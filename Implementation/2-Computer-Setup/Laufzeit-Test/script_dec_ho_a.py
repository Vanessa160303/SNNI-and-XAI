# PyTorch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist

# CrypTen
import crypten
import crypten.mpc as mpc

# Third-Party Libraries
import numpy as np
import matplotlib.pyplot as plt

# Standard Libraries
import argparse
import os
import time
from statistics import mean, stdev, mean

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

def main():
    """
    Performs the occlusion analysis with the hierarchical occlusion analysis algorithm on Alice-Net.
    Parties send their unencrypted data/model to communicate.
    The resulting heatmap is saved as a picture.
    """

    # Parsing given arguments for the communication
    parser = argparse.ArgumentParser()
    parser.add_argument("--rank", type=int, required=True) # What is the rank of the party on the machine
    parser.add_argument("--world_size", type=int, default=2) # What is the current world size (how many parties are there)
    parser.add_argument("--master_addr", type=str, default="localhost") # What is the address of one of the ranks (mainly rank 0)
    args = parser.parse_args()

    dist.init_process_group(
        backend='gloo', # To Change
        init_method='tcp://', # To Change
        rank=args.rank, 
        world_size=2 # To Change
    )

    if args.rank == 0:  # Rank of Alice: Owner of model
        run_alice()
    else:  # Rank of Bob: Owns the data
        run_bob()

    dist.destroy_process_group()  # Group needs to be destroyed in the end  

def run_alice():
    """Runs the occlusion analysis with Bob's data"""
    print("Alice: Starting occlusion analysis")
    
    # Load model
    n = 10  # Number of repetitions, can be changed

    # Arrays to store the calculated metrics 
    all_load_times = []
    all_prep2_times = []
    all_vis_times = []

    for run in range(n):
        print(f"\nCurrent run {run+1}/{n}")
    
        # Start time for loading
        load_time_start = time.time()*1000
        model = torch.load('models/tutorial4_alice_model.pth', weights_only=False)
        model.eval() # Put model in inference phase

        # End time for loading
        load_time_end = time.time()*1000

        # Calculate the time of the loading
        load_time = (load_time_end - load_time_start)

        # Put in the calculated metrics
        all_load_times.append(load_time)

        # Start time for prep2
        prep2_time_start = time.time()*1000

        # Get data from Bob
        data = torch.zeros(1, 28, 28)  # Placeholder
        dist.recv(tensor=data, src=1)
        print("Alice: Received data from Bob")

        # End time for prep2
        prep2_time_end = time.time()*1000

        # Calculate the time of the prep2
        prep2_time = (prep2_time_end - prep2_time_start)

        # Put in the calculated metrics
        all_prep2_times.append(prep2_time)

        # Actual Occlusion Analysis
        heatmap = perform_occlusion_analysis(model, data[0], run+1)

        # Start time for for visualization
        vis_time_start = time.time()*1000
        heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min())
        
        # Send heatmap to Bob
        dist.send(tensor=heatmap, dst = 1)
        print("Alice: Send heatmap to Bob")
        
        # Save results as a picture
        plt.imshow(heatmap, cmap='hot')
        plt.savefig('heatmap_dec_ho.png')
        print("Alice: Analysis completed and saved")

        # End time for visualization
        vis_time_end = time.time()*1000
    
        # Calculate the time of the visualization
        vis_time = vis_time_end - vis_time_start

        # Put in the calculated metrics
        all_vis_times.append(vis_time)

    # Print metrics
    print(f"Time in ms (Load): Mean={mean(all_load_times):.3f}, Stdev={stdev(all_load_times):.3f}, Min={min(all_load_times):.3f}, Max={max(all_load_times):.3f}")
    print(f"Time in ms (Prep2): Mean={mean(all_prep2_times):.3f}, Stdev={stdev(all_prep2_times):.3f}, Min={min(all_prep2_times):.3f}, Max={max(all_prep2_times):.3f}")
    print(f"Time in ms (Vis): Mean={mean(all_vis_times):.3f}, Stdev={stdev(all_vis_times):.3f}, Min={min(all_vis_times):.3f}, Max={max(all_vis_times):.3f}")
    
def run_bob():
    """Send data to Alice"""
    print("Bob: Preparing data")
    
    # Load data
    n = 10  # Number of repetitions, can be changed

    # Arrays to store the calculated metrics
    all_prep1_times = []
    all_vis1_times = []
   
    for run in range(n):
        print(f"\nCurrent run {run+1}/{n}")
    
        # Start time for prep1
        prep1_time_start = time.time()*1000
        data = torch.load('/tmp/bob_test.pth')[0].unsqueeze(0) # Unsqueeze was needed to add another dimension
        image = data.view(28, 28)

        # Send data to Alice
        dist.send(tensor=image, dst=0)
        print("Bob: Data sent to Alice")

        # End time for prep1
        prep1_time_end = time.time()*1000

        # Calculate the time of the prep1
        prep1_time = (prep1_time_end - prep1_time_start)

        # Put in the calculated metrics
        all_prep1_times.append(prep1_time)

        # Start time for for visualization
        vis1_time_start = time.time()*1000

        # Get heatmap from Alice
        heatmap = torch.zeros((28,28)) #Placeholder
        dist.recv(tensor=heatmap, src=0)
        print("Bob:Got Heatmap")
        
        # Save results
        plt.imshow(heatmap, cmap='hot')
        plt.savefig('heatmap_dec_ho.png')
        # End time for visualization
        vis1_time_end = time.time()*1000
    
        # Calculate the time of the visualization
        vis1_time = vis1_time_end - vis1_time_start

        all_vis1_times.append(vis1_time)

    # Print metrics
    print(f"Time in ms (Prep1): Mean={mean(all_prep1_times):.3f}, Stdev={stdev(all_prep1_times):.3f}, Min={min(all_prep1_times):.3f}, Max={max(all_prep1_times):.3f}")
    print(f"Time in ms (Vis1): Mean={mean(all_vis1_times):.3f}, Stdev={stdev(all_vis1_times):.3f}, Min={min(all_vis1_times):.3f}, Max={max(all_vis1_times):.3f}")
    
def perform_occlusion_analysis(model, image, n1):
    """Does actual Occlusion Analysis"""
    n = 10  # Number of repetitions, can be changed

    # Arrays to store the calculated metrics for the different phases
    all_prep_times = []    
    all_occl_times = []  
    all_iteration_counts = []

    for run in range(n):
        print(f"\nCurrent run {run+1}/{n}")
        # Arrays for the Occlusion Analysis per iteration
        run_occl_times = []

        # Start time for prep
        prep_time_start = time.time()*1000

        # Parameters for Occlusion Analysis 
        # - initial_patch
        # - minimal_patch
        # - occluded_value
        # - threshold
        # - heatmap
        initial_patch = 8 # Can be changed
        minimal_patch = 4 # Can be changed
        occluded_value = 0 # Can be changed
        threshold = 0.001 # Can be changed
        heatmap = torch.zeros((image.shape[0], image.shape[1]))

        # Computation of original output of the original picture
        with torch.no_grad(): # Allows quicker computation
            original_output = model(image.view(1, 784)) # View is needed for the input of the model

        # End time for prep
        prep_time_end = time.time()*1000
    
        # Calculate the time of the prep
        prep_time = (prep_time_end - prep_time_start)

        num_it = 1 # Counter for number of iterations
        
        def patch_partition(starting_point, current_patch_size, heatmap):
            """
            Partions the patch if the differenc of the original output and occluded output is important.
            Args:
                starting_point: top-left corner of the current patch
                current_patch_size: Size of the current patch
                heatmap: Contains the difference of the original output and occluded output
            Returns:
                heatmap
            """  

            # Start time for Occlusion Analysis
            occl_time_start = time.time()*1000

            nonlocal num_it
            y, x = starting_point

            # Edge treatment:
            # The normal patch size is used 
            # Unless the patch doesn't entirely fit in the edge
            # Then the patch_size is the image.size - the current pixel position
            better_patch_size_y = min(current_patch_size, image.shape[0] - y)
            better_patch_size_x = min(current_patch_size, image.shape[1] - x)

            # Used if patch would be too small
            if better_patch_size_y < 1 or better_patch_size_x < 1:
                return heatmap      

            # Clone the picture and put the mask on the position
            occluded_image = image.clone()
            occluded_image[y : y + better_patch_size_y, x : x + better_patch_size_x] = occluded_value

            # Calculating the occluded_output
            with torch.no_grad():
                occluded_output = model(occluded_image.view(1, 784))
                
            # Calculating the absolute difference of the original output and the occluded output
            diff = (original_output - occluded_output).abs().sum().item()
            
            # Put the difference in the heatmap
            heatmap[y : y + better_patch_size_y, x : x + better_patch_size_x] += diff

            # Calculates the difference but in the range 0 to 1
            comp = ((original_output.softmax(1) - occluded_output.softmax(1)).abs().sum())/ 2

            # Decision if the patch should be divided
            # Difference needs to be bigger than threshold 
            # and patch must be bigger than smallest patch
            differentiate = (comp > threshold) and (float(current_patch_size)/2.0 >= float(minimal_patch))

            # End time for Occlusion Analysis
            occl_time_end = time.time()*1000
    
            # Calculate the time of the Occlusion Analysis
            occl_time = occl_time_end - occl_time_start

            # Put in the calculated metrics for the current iteration
            run_occl_times.append(occl_time)
            
            # If decision is true, divide the patch 
            # Set the position for the patches
            if differentiate:
                half_patch = current_patch_size // 2
                # Calculate possible positions and set new positions
                for poss_y in [0, half_patch]:
                    for poss_x in [0, half_patch]:
                        new_y = poss_y + y
                        new_x = poss_x + x
                        # Check if patch position is bigger than picture
                        if (new_y < image.shape[0] and new_x < image.shape[1]):
                            heatmap = patch_partition((new_y, new_x), half_patch, heatmap)
            
            print(f"\rNumber of iterations: {num_it}", end='', flush=True)
            num_it += 1
            
            return heatmap
        
        # Start of Occlusion Analysis
        for y in range(0, image.shape[0], initial_patch):
            for x in range(0, image.shape[1], initial_patch):
                heatmap = patch_partition((y,x), initial_patch, heatmap) 

        # Store iteration count for this run
        all_iteration_counts.append(num_it - 1)    

        # Add the values for the run       
        all_prep_times.append(prep_time)        
        all_occl_times.append(run_occl_times)

    # Print metrics        
    if n1 == n:
        prep_time_stats = calculate_stats(all_prep_times)
        occl_time_stats = calculate_stats([item for sublist in all_occl_times for item in sublist])
    
        print(f"Time in ms (Prep): Mean={prep_time_stats['mean']:.3f}, Stdev={prep_time_stats['stdev']:.3f}, Min={prep_time_stats['min']:.3f}, Max={prep_time_stats['max']:.3f}")
        print(f"Time in ms (Occl): Mean={occl_time_stats['mean']:.3f}, Stdev={occl_time_stats['stdev']:.3f}, Min={occl_time_stats['min']:.3f}, Max={occl_time_stats['max']:.3f}, Total={occl_time_stats['total']}")

    return heatmap

# Def to help with calculation
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
        
if __name__ == '__main__':
    main()