#!/usr/bin/env python3

# PyTorch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist

# CrypTen
import crypten
import crypten.mpc as mpc
import crypten.communicator as comm
from crypten.config import cfg

# Standard Libraries
import argparse
import os
import time
from statistics import median, stdev, mean

# Third-Party Libraries
import numpy as np
import matplotlib.pyplot as plt

# Parsing given arguments for the communication
parser = argparse.ArgumentParser()
parser.add_argument("--rank", type=int, required=True) # What is the rank of the party on the machine
parser.add_argument("--world_size", type=int, default=2) # What is the current world size (how many parties are there)
parser.add_argument("--master_addr", type=str, default="localhost") # What is the address of one of the ranks (mainly rank 0)
args = parser.parse_args()

# Initializing CrypTen
cfg.communicator.verbose = True
crypten.init() 

crypten.print("Crypten is init")

# Define Alice's network (from CrypTen tutorials)
class AliceNet(torch.nn.Module):
    def __init__(self):
        super(AliceNet, self).__init__()
        self.fc1 = torch.nn.Linear(784, 128)
        self.fc2 = torch.nn.Linear(128, 128)
        self.fc3 = torch.nn.Linear(128, 10)
    
    def forward(self, x):
        x = torch.nn.functional.relu(self.fc1(x))
        x = torch.nn.functional.relu(self.fc2(x))
        return self.fc3(x)

# Definition of the two parties
ALICE, BOB = 0, 1

def run_script():
    """
    Performs the occlusion analysis with the monte-carlo algorithm on Alice-Net.
    Parties send their encrypted data/model to communicate.
    The resulting heatmap is saved as a picture.
    """
    n = 3  # Number of repetitions, can be changed
    # Arrays to store the calculated metrics for the different phases
    all_load_times = []
    
    all_prep_times = []
    all_prep_com_rounds = []
    all_prep_com_bytes = []
    all_prep_com_time = []
    
    all_occl_times = []  
    all_occl_com_rounds = []
    all_occl_com_bytes = []
    all_occl_com_time = []
    
    all_vis_times = []
    all_vis_com_rounds = []
    all_vis_com_bytes = []
    all_vis_com_time = []
    
    all_iteration_counts = []

    for run in range(n):
        crypten.print(f"\nCurrent run {run+1}/{n}")
        # Start time for loading
        load_time_start = time.time()*1000

        # Load model and convert it(from CrypTen tutorials)
        # Encrypt the model from Alice (from CrypTen tutorial)
        model = model = torch.load('models/tutorial4_alice_model.pth', weights_only = False)
        dummy_input = torch.empty((1, 784))
        private_model = crypten.nn.from_pytorch(model, dummy_input)

        # End time for loading
        load_time_end = time.time()*1000

        # Calculate the time of the loading
        load_time = (load_time_end - load_time_start)

        # Put in the calculated metrics
        all_load_times.append(load_time)

        # Arrays for the Occlusion Analysis per iteration
        run_occl_times = []
        run_occl_com_rounds = []
        run_occl_com_bytes = []
        run_occl_com_time = []

        comm.get().reset_communication_stats()
    
        # Start time for prep
        prep_time_start = time.time()*1000

        private_model.encrypt(src = 0)
        private_model.eval()
        crypten.print("Alice: Encrypted model")

        # Get encrypted data from Bob
        data_enc = crypten.load_from_party('/tmp/bob_test.pth', src = 1)
        data_enc = data_enc[0].unsqueeze(0)
        crypten.print("Bob: Encrypted data and send data")

        # Parameters for Occlusion Analysis 
        # - patch_size
        # - occluded_value
        # - num_simulations
        # - encrypted heatmap
        patch_size = 5 # Can be changed
        occluded_value = 0 # Can be changed
        num_simulations = 100 # Can be changed
        heatmap_enc = crypten.cryptensor(torch.zeros(data_enc.shape[1], data_enc.shape[2]), src = 1)

        # Computation of original output of the original picture
        with torch.no_grad(): # Allows quicker computation
            original_output = private_model(data_enc.view(1, 784)) # View is needed for the input of the model

        comm.get().reset_communication_stats()

        num_it = 1 # Counter for number of iterations

        # Actual_sim represents the number of actual simulations
        # Tried_sim is the number of tried simulations (those who failed and those who didn't
        # Max_sim is the number of maximum tries to avoid endless loop
        actual_sim = 0
        tried_sim = 0
        max_sim = num_simulations * 2 # Can be changed

        # Calculate the actual possible positions for the patch
        # By dividing the possible positions by the patch_size
        poss_positions_y = (data_enc.shape[1] - patch_size) // patch_size + 1
        poss_positions_x = (data_enc.shape[2] - patch_size) // patch_size + 1

        # End time for prep
        prep_time_end = time.time()*1000
    
        # Calculate the time of the prep
        prep_time = (prep_time_end - prep_time_start)
    
        # Get the communication stats and reset them afterwards
        rounds_com_prep = comm.get().comm_rounds
        bytes_com_prep = comm.get().comm_bytes
        time_com_prep = comm.get().comm_time

        # Add the values of the communication before the reset
        all_prep_com_rounds.append(rounds_com_prep)
        all_prep_com_bytes.append(bytes_com_prep)
        all_prep_com_time.append(time_com_prep)
        
        comm.get().reset_communication_stats()

        while actual_sim < num_simulations and tried_sim < max_sim:
            # Start time for Occlusion Analysis
            occl_time_start = time.time()*1000

            # Choose a position for the patch
            position_patch_y = np.random.randint(0, poss_positions_y + 1)
            position_patch_x = np.random.randint(0, poss_positions_x + 1)

            # Calculate the positions with the patch_size
            y = position_patch_y * patch_size
            x = position_patch_x * patch_size

            # Edge treatment:
            # The normal patch size is used 
            # Unless the patch doesn't entirely fit in the edge
            # Then the patch_size is the image.size - the current pixel position
            better_patch_size_y = min(patch_size, data_enc.shape[1] - y)
            better_patch_size_x = min(patch_size, data_enc.shape[2] - x)

            # Used if patch would be too small
            if better_patch_size_y < 1 or better_patch_size_x < 1:
                    tried_sim += 1
                    continue

            # Creating a mask the size of the picture with value one
            # Marking the relevant place, setting it to the occluded value
            # Encrypting the mask
            mask = torch.ones(data_enc.shape[0], data_enc.shape[1], data_enc.shape[2])
            mask[: , y : y + better_patch_size_y, x : x + better_patch_size_x] = occluded_value
            mask_enc = crypten.cryptensor(mask, src = 1)

            # Putting the mask on the picture
            occluded_enc = data_enc * mask_enc

            # Calculating the occluded_output
            with torch.no_grad():
                occluded_output = private_model(occluded_enc.view(1, 784))

            # Calculating the absolute difference of the original output and the occluded output
            diff_enc = (original_output - occluded_output).abs().sum()

            # Putting the calculated difference on the encrypted heatmap
            # By creating a patch that has value one on the relevant region
            # Only where the value is one the diff will be added to the heatmap
            place_heatmap = torch.zeros(heatmap_enc.shape[0], heatmap_enc.shape[1])
            place_heatmap[y : y  + better_patch_size_y, x : x + better_patch_size_x] = 1
            heatmap_enc = heatmap_enc + diff_enc * crypten.cryptensor(place_heatmap, src = 1)

            # End time for Occlusion Analysis
            occl_time_end = time.time()*1000

            # Calculate the time of the Occlusion Analysis
            occl_time = occl_time_end - occl_time_start

            # Get the communication stats and reset them afterwards
            rounds_com_occl = comm.get().comm_rounds
            bytes_com_occl = comm.get().comm_bytes
            time_com_occl = comm.get().comm_time

            # Add the values of the communication before the reset
            run_occl_com_rounds.append(rounds_com_occl)
            run_occl_com_bytes.append(bytes_com_occl)
            run_occl_com_time.append(time_com_occl)
            
            comm.get().reset_communication_stats()

            # Put in the calculated metrics for the current iteration
            run_occl_times.append(occl_time)

            crypten.print(f"\rNumber of iterations: {num_it}", end='', flush=True)
            num_it += 1
            tried_sim += 1
            actual_sim += 1

        # Store iteration count for this run
        all_iteration_counts.append(num_it - 1) 
    
        # Start time for for visualization
        vis_time_start = time.time()*1000

        # Calculate the average influence
        heatmap = heatmap_enc.get_plain_text()/ actual_sim

        # Normalizing the heatmap
        heatmap_norm = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min())

        # Saving the heatmap as a picture
        plt.imshow(heatmap_norm, cmap='hot')
        plt.savefig('heatmap_mc.png')

        # End time for visualization
        vis_time_end = time.time()*1000
    
        # Calculate the time of the visualization
        vis_time = vis_time_end - vis_time_start
    
        # Get the communication stats and reset them afterwards
        rounds_com_vis = comm.get().comm_rounds
        bytes_com_vis = comm.get().comm_bytes
        time_com_vis = comm.get().comm_time

        # Add the values of the communication before the reset
        all_vis_com_rounds.append(rounds_com_vis)
        all_vis_com_bytes.append(bytes_com_vis)
        all_vis_com_time.append(time_com_vis)
        
        comm.get().reset_communication_stats()
    
        # Add the values for the run
        all_prep_times.append(prep_time)
        
        all_occl_times.append(run_occl_times)
        all_occl_com_rounds.append(run_occl_com_rounds)
        all_occl_com_bytes.append(run_occl_com_bytes)
        all_occl_com_time.append(run_occl_com_time)
        
        all_vis_times.append(vis_time)

        # Only after the last run the results should be presented and calculated
        if run + 1 == n:
            # Calculate mean, std, min and max for given data
            def calculate_stats(data):

                # Checking if the array has more then one dimension
                if isinstance(data[0], list): 
                    flat_data = [item for sublist in data for item in sublist]
                    return {
                        'mean': mean(flat_data),
                        'stdev': stdev(flat_data) if len(flat_data) > 1 else 0,
                        'min': min(flat_data),
                        'max': max(flat_data),
                        'total': sum(flat_data)
                    }
                else:
                    return {
                        'mean': mean(data),
                        'stdev': stdev(data) if len(data) > 1 else 0,
                        'min': min(data),
                        'max': max(data),
                        'total': sum(data)
                    }
            
            # Calculate the values for every metric
            load_time_stats = calculate_stats(all_load_times)

            prep_time_stats = calculate_stats(all_prep_times)
            prep_com_rounds_stats = calculate_stats(all_prep_com_rounds)
            prep_com_bytes_stats = calculate_stats(all_prep_com_bytes)
            prep_com_time_stats = calculate_stats(all_prep_com_time)
            
            occl_time_stats = calculate_stats([item for sublist in all_occl_times for item in sublist])
            occl_com_rounds_stats = calculate_stats([item for sublist in all_occl_com_rounds for item in sublist])
            occl_com_bytes_stats = calculate_stats([item for sublist in all_occl_com_bytes for item in sublist])
            occl_com_time_stats = calculate_stats([item for sublist in all_occl_com_time for item in sublist])
            
            vis_time_stats = calculate_stats(all_vis_times)
            vis_com_rounds_stats = calculate_stats(all_vis_com_rounds)
            vis_com_bytes_stats = calculate_stats(all_vis_com_bytes)
            vis_com_time_stats = calculate_stats(all_vis_com_time)
            
            # Show the calculated statistics
            crypten.print("\n\n==Values of the test==")
            crypten.print(f"\nIterations per run: {sum(all_iteration_counts)/n:.1f}")
            crypten.print(f"\nTimes the code was tested: {n}")

            crypten.print("\nLoad:")
            crypten.print(f"Time in ms: Mean={load_time_stats['mean']:.3f}, Stdev={load_time_stats['stdev']:.3f}, Min={load_time_stats['min']:.3f}, Max={load_time_stats['max']:.3f}")

            crypten.print("\nPrep:")
            crypten.print(f"Time in ms: Mean={prep_time_stats['mean']:.3f}, Stdev={prep_time_stats['stdev']:.3f}, Min={prep_time_stats['min']:.3f}, Max={prep_time_stats['max']:.3f}")
            crypten.print(f"Comm Rounds: Mean={prep_com_rounds_stats['mean']}, Stdev={prep_com_rounds_stats['stdev']:.1f}, Total={prep_com_rounds_stats['total']}")
            crypten.print(f"Comm Bytes: Mean={prep_com_bytes_stats['mean']}, Stdev={prep_com_bytes_stats['stdev']:.1f}, Total={prep_com_bytes_stats['total']}")
            crypten.print(f"Comm Time in s: Mean={prep_com_time_stats['mean']:.3f}, Stdev={prep_com_time_stats['stdev']:.3f}, Total={prep_com_time_stats['total']:.3f}")
            
            crypten.print("\nActual Occlusion Analysis:")
            crypten.print(f"Time in ms: Mean={occl_time_stats['mean']:.3f}, Stdev={occl_time_stats['stdev']:.3f}, Min={occl_time_stats['min']:.3f}, Max={occl_time_stats['max']:.3f}, Total={occl_time_stats['total']}")
            crypten.print(f"Comm Rounds: Mean={occl_com_rounds_stats['mean']}, Stdev={occl_com_rounds_stats['stdev']:.1f}, Total={occl_com_rounds_stats['total']}")
            crypten.print(f"Comm Bytes: Mean={occl_com_bytes_stats['mean']}, Stdev={occl_com_bytes_stats['stdev']:.1f}, Total={occl_com_bytes_stats['total']}")
            crypten.print(f"Comm Time in s: Mean={occl_com_time_stats['mean']:.3f}, Stdev={occl_com_time_stats['stdev']:.3f}, Total={occl_com_time_stats['total']:.3f}")
            
            crypten.print("\nVisualization:")
            crypten.print(f"Time in ms: Mean={vis_time_stats['mean']:.3f}, Stdev={vis_time_stats['stdev']:.3f}, Min={vis_time_stats['min']:.3f}, Max={vis_time_stats['max']:.3f}")
            crypten.print(f"Comm Rounds: Mean={vis_com_rounds_stats['mean']}, Stdev={vis_com_rounds_stats['stdev']:.1f}, Total={vis_com_rounds_stats['total']}")
            crypten.print(f"Comm Bytes: Mean={vis_com_bytes_stats['mean']}, Stdev={vis_com_bytes_stats['stdev']:.1f}, Total={vis_com_bytes_stats['total']}")
            crypten.print(f"Comm Time in s: Mean={vis_com_time_stats['mean']:.3f}, Stdev={vis_com_time_stats['stdev']:.3f}, Total={vis_com_time_stats['total']:.3f}")

def main():
    """ Runs the script """
    run_script() 
    dist.destroy_process_group() # Group needs to be destroyed in the end

if __name__ == '__main__':
    main()