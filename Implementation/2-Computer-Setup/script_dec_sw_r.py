# PyTorch and Torchvision
import torch
import torchvision.transforms as transforms
import torchvision.models as models
import torch.distributed as dist

# Standard Libraries
import json
import argparse
import os

# Third-Party Libraries
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import requests

# CrypTen
import crypten
import crypten.mpc as mpc

def process_image(actual_path, save_path):
    """ 
    Takes the original picture and prepares it for ResNet18
    Matches the size to 224x 224, transforms it to a tensor and normalizes it with recommended means and std for every colour channel
    Saves the picture in the temp file
    """
    # Actual transformation
    transform_pic = transforms.Compose([transforms.Resize((224, 224)),transforms.ToTensor(),transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    with Image.open(actual_path) as img:
        img_processed = transform_pic(img.convert("RGB")).unsqueeze(0) 
    
    # Save picture
    torch.save(img_processed, save_path)
    
process_image("test_image.jpg","/tmp/processed_image_dec.pth")

def main():
    """
    Performs the occlusion analysis with the sliding-window algorithm on ResNet18.
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

    if args.rank == 0:  # Rank of Alice: Owner fo model
        run_alice()
    else:  # Rank of Bob: Owns the data
        run_bob()

    dist.destroy_process_group()  # Group needs to be destroyed in the end  

def run_alice():
    """Runs the occlusion analysis with Bob's data"""
    print("Alice: Starting occlusion analysis")
    
    # Load model
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1) # Pre-trained version of ResNet18
    model.eval() # Put model in inference phase
    
    # Get data from Bob
    data = torch.zeros(1, 3, 224, 224)  # Placeholder
    dist.recv(tensor=data, src=1)
    print("Alice: Received data from Bob")
    
    # Actual Occlusion Analysis
    heatmap = perform_occlusion_analysis(model, data)
    
    # Send heatmap to Bob
    dist.send(tensor=heatmap, dst = 1)
    print("Alice: Send heatmap to Bob")
    
    # Save results as a picture
    plt.imshow(heatmap, cmap='hot')
    plt.savefig('heatmap_dec_sw.png')
    print("Alice: Analysis completed and saved")

def run_bob():
    """Send data to Alice"""
    print("Bob: Preparing data")
    
    # Load data
    image = torch.load('/tmp/processed_image_dec.pth') 
    
    # Send data to Alice
    dist.send(tensor=image, dst=0)
    print("Bob: Data sent to Alice")
    
    # Get heatmap from Alice
    heatmap = torch.zeros((224,224)) #Placeholder
    dist.recv(tensor=heatmap, src=0)
    print("Bob:Got Heatmap")
    
    # Save results
    plt.imshow(heatmap, cmap='hot')
    plt.savefig('heatmap_dec_sw.png')

def perform_occlusion_analysis(model, image):
    """Does actual Occlusion Analysis"""
    # Parameters for Occlusion Analysis 
    # - patch size
    # - occluded_value
    # - heatmap
    patch_size = 112 # Can be changed
    occluded_value = 0 # Can be changed
    heatmap = torch.zeros((image.shape[2], image.shape[3]))

    # Computation of original output of the original picture
    with torch.no_grad(): # Allows quicker computation
        original_output = model(image)

    num_it = 1 # Counter for number of iterations

    # Iterating over the picture
    for y in range(0, image.shape[2], patch_size):
        for x in range(0, image.shape[3], patch_size):
            # Edge treatment:
            # The normal patch size is used 
            # Unless the patch doesn't entirely fit in the edge
            # Then the patch_size is the image.size - the current pixel position
            better_patch_size_y = min(patch_size, image.shape[2] - y)
            better_patch_size_x = min(patch_size, image.shape[3] - x)
            
            # Used if patch would be too small
            if better_patch_size_y < 1 or better_patch_size_x < 1:
                continue
                
            # Clone the picture and put the mask on the position
            occluded_image = image.clone()
            occluded_image[:, :, y : y + better_patch_size_y, x : x + better_patch_size_x] = occluded_value

            # Calculating the occluded_output
            with torch.no_grad():
                occluded_output = model(occluded_image)
            
            # Calculating the absolute difference of the original output and the occluded output
            diff = (original_output - occluded_output).abs().sum().item()

            # Put the difference in the heatmap
            heatmap[y : y + better_patch_size_y, x : x + better_patch_size_x] += diff

            print(f"\rNumber of iterations: {num_it}", end='', flush=True)
            num_it += 1

    # Normalizing the heatmap
    return (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min())
    
if __name__ == '__main__':
    main()

