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

# Third-Party Libraries
import numpy as np
import argparse
import os
import matplotlib.pyplot as plt

# Parsing given arguments for the communication
parser = argparse.ArgumentParser()
parser.add_argument("--rank", type=int, required=True) # What is the rank of the party on the machine
parser.add_argument("--world_size", type=int, default=2) # What is the current world size (how many parties are there)
parser.add_argument("--master_addr", type=str, default="localhost") # What is the address of one of the ranks (mainly rank 0)
args = parser.parse_args()

# Initializing CrypTen
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
    Performs the occlusion analysis with the sliding-window algorithm on Alice-Net.
    Parties send their encrypted data/model to communicate.
    The resulting heatmap is saved as a picture.
    """
    # Load model and convert it(from CrypTen tutorials)
    # Encrypt the model from Alice (from CrypTen tutorial)
    model = model = torch.load('models/tutorial4_alice_model.pth', weights_only = False)
    dummy_input = torch.empty((1, 784))
    private_model = crypten.nn.from_pytorch(model, dummy_input)
    private_model.encrypt(src = 0)
    private_model.eval()
    crypten.print("Alice: Encrypted model")

    # Get encrypted data from Bob
    data_enc = crypten.load_from_party('/tmp/bob_test.pth', src = 1)
    data_enc = data_enc[0].unsqueeze(0)
    crypten.print("Bob: Encrypted data and send data")
    
    # Parameters for Occlusion Analysis 
    # - patch size
    # - occluded_value
    # - encrypted heatmap
    patch_size = 4 # Can be changed
    occluded_value = 0 # Can be changed
    heatmap_enc = crypten.cryptensor(torch.zeros(data_enc.shape[1], data_enc.shape[2]), src = 1)

    # Computation of original output of the original picture
    with torch.no_grad(): # Allows quicker computation
        original_output = private_model(data_enc.view(1, 784)) # View is needed for the input of the model
    
    num_it = 1 # Counter for number of iterations

    # Iterating over the picture
    for y in range(0, data_enc.shape[1], patch_size):
        for x in range(0, data_enc.shape[2], patch_size):
            # Edge treatment:
            # The normal patch size is used 
            # Unless the patch doesn't entirely fit in the edge
            # Then the patch_size is the image.size - the current pixel position
            better_patch_size_y = min(patch_size, data_enc.shape[1] - y)
            better_patch_size_x = min(patch_size, data_enc.shape[2] - x)
            
            # Used if patch would be too small
            if better_patch_size_y < 1 or better_patch_size_x < 1:
                continue

            # Creating a mask the size of the picture with value one
            # Marking the relevant place, setting it to the occluded value
            # Encrypting the mask
            mask = torch.ones(data_enc.shape[0], data_enc.shape[1], data_enc.shape[2])
            mask[:, y : y + better_patch_size_y, x : x + better_patch_size_x] = occluded_value
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

            crypten.print(f"\rNumber of iterations: {num_it}", end='', flush=True)
            num_it += 1

    # Decrypting the heatmap 
    heatmap = heatmap_enc.get_plain_text()

    # Normalizing the heatmap
    heatmap_norm = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min())
    
    # Saving the heatmap as a picture
    plt.imshow(heatmap_norm, cmap='hot')
    plt.savefig('heatmap_sw.png')

def main():
    """ Runs the script """
    run_script() 
    dist.destroy_process_group() # Group needs to be destroyed in the end

if __name__ == '__main__':
    main()

