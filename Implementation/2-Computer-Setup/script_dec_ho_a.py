# PyTorch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist

# CrypTen
import crypten
import crypten.mpc as mpc

# Standard libraries
import argparse
import os

# Third-Party Libraries
import numpy as np
import matplotlib.pyplot as plt

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
    model = torch.load('models/tutorial4_alice_model.pth', weights_only=False)
    model.eval() # Put model in inference phase
    
    # Get data from Bob
    data = torch.zeros(1, 28, 28)  # Placeholder
    dist.recv(tensor=data, src=1)
    print("Alice: Received data from Bob")
    
    # Actual Occlusion Analysis
    heatmap = perform_occlusion_analysis(model, data[0])
    
    # Send heatmap to Bob
    dist.send(tensor=heatmap, dst = 1)
    print("Alice: Send heatmap to Bob")
    
    # Save results as a picture
    plt.imshow(heatmap, cmap='hot')
    plt.savefig('heatmap_dec_ho.png')
    print("Alice: Analysis completed and saved")

def run_bob():
    """Send data to Alice"""
    print("Bob: Preparing data")
    
    # Load data
    data = torch.load('/tmp/bob_test.pth')[0].unsqueeze(0) # Unsqueeze was needed to add another dimension
    image = data.view(28, 28)
    
    # Send data to Alice
    dist.send(tensor=image, dst=0)
    print("Bob: Data sent to Alice")
    
    # Get heatmap from Alice
    heatmap = torch.zeros((28,28)) #Placeholder
    dist.recv(tensor=heatmap, src=0)
    print("Bob:Got Heatmap")
    
    # Save results
    plt.imshow(heatmap, cmap='hot')
    plt.savefig('heatmap_dec_ho.png')

def perform_occlusion_analysis(model, image):
    """Does actual Occlusion Analysis"""
    # Parameters for Occlusion Analysis 
    # - initial_patch
    # - minimal_patch
    # - occluded_value
    # - threshold
    # - heatmap
    initial_patch = 6 # Can be changed
    minimal_patch = 2 # Can be changed
    occluded_value = 0 # Can be changed
    threshold = 0.001 # Can be changed
    heatmap = torch.zeros((image.shape[0], image.shape[1]))

    # Computation of original output of the original picture
    with torch.no_grad(): # Allows quicker computation
        original_output = model(image.view(1, 784)) # View is needed for the input of the model

    num_it = 1 # Counter for number of iterations
    
    def patch_partition(starting_point, current_patch_size, heatmap):
        """
        Partitions the patch if the difference of the original output and occluded output is important.
        Args:
            starting_point: top-left corner of the current patch
            current_patch_size: Size of the current patch
            heatmap: Contains the difference of the original output and occluded output
        Returns:
            heatmap
        """  
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
            
    # Normalizing the heatmap
    return (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min())
    
if __name__ == '__main__':
    main()