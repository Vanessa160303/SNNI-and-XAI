# PyTorch and Torchvision
import torch
import torchvision.models as models
import torchvision.transforms as transforms
import torch.distributed as dist

# Standard Libraries
from PIL import Image
import json
import requests

# Third-Party Libraries
import numpy as np
import matplotlib.pyplot as plt
import onnx
import argparse
import os

# CrypTen
import crypten
import crypten.mpc as mpc
from crypten.nn import onnx_converter
import crypten.communicator as comm

# Parsing given arguments for the communication
parser = argparse.ArgumentParser()
parser.add_argument("--rank", type=int, required=True) # What is the rank of the party on the machine
parser.add_argument("--world_size", type=int, default=2) # What is the current world size (how many parties are there)
parser.add_argument("--master_addr", type=str, default="localhost") # What is the address of one of the ranks (mainly rank 0)
args = parser.parse_args()

# Initializing CrypTen
crypten.init() 

crypten.print("Crypten is init")

# Definition of the two parties
ALICE, BOB = 0, 1

def prepare_resnet18():
    """
    Takes the ResNet18 model and converts it into a ONNX-model
    Skips Identity-Operations that cause errors
    Saves the model to a file so it can be loaded and encrypted later
    """
    # The ResNet18 model
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    model.eval() # Put model in inference phase

    # Dummy Input for ResNet18 
    dummy_input = torch.randn(1, 3, 224, 224)  

    # Actual conversion from pytorch to ONNX
    onnx_path="resnet18.onnx"
    torch.onnx.export(
        model,
        dummy_input,
        "resnet18.onnx",
        input_names = ["input"],
        output_names = ["output"],
        do_constant_folding = True, #For optimization
        opset_version = 9,  #Supported Version for CrypTen
        dynamic_axes = {"input": {0: "batch"}},  
        
    )
    # Reads ONNX-file as a byte stream
    with open(onnx_path, "rb") as f:
        onnx_model_bytes = f.read()
    return onnx_model_bytes
#onnx_model_bytes = prepare_resnet18()
crypten.print("Model is prepared")

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
crypten.print("Picture is saved")

def run_script(onnx_model_bytes):
    """
    Performs the occlusion analysis with the hierarchical occlusion analysis algorithm on ResNet18.
    Parties send their encrypted data/model to communicate.
    The resulting heatmap is saved as a picture.
    """
    # Load model an convert it
    private_model = crypten.nn.from_onnx(onnx_model_bytes)
    # Encrypt the model from Alice
    private_model.encrypt(src = 0)
    crypten.print("Alice: Encrypted model")

    # Load the picture and encrypt it with Bob
    data_enc = crypten.load_from_party('/tmp/processed_image.pth', src = 1)
    crypten.print("Bob: Encrypted data and send data")
    # Parameters for Occlusion Analysis 
    # - initial_patch
    # - minimal_patch
    # - occluded_value
    # - encrypted threshold
    # - encrypted heatmap
    initial_patch = 112 # Can be changed
    minimal_patch = 112 # Can be changed
    occluded_value = 0 # Can be changed
    threshold_enc = crypten.cryptensor(0.5, src = 1) # Can be changed
    heatmap_enc = crypten.cryptensor(torch.zeros(data_enc.shape[2], data_enc.shape[3]), src = 1)

    # Computation of original output of the original picture
    with torch.no_grad(): # Allows quicker computation
        original_output = private_model(data_enc)

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
        better_patch_size_y = min(current_patch_size, data_enc.shape[2] - y)
        better_patch_size_x = min(current_patch_size, data_enc.shape[3] - x)

        # Used if patch would be too small
        if better_patch_size_y < 1 or better_patch_size_x < 1:
            return heatmap      
              
        # Creating a mask the size of the picture with value one
        # Marking the relevant place, setting it to the occluded value
        # Encrypting the mask
        mask = torch.ones(data_enc.shape[0], data_enc.shape[1], data_enc.shape[2], data_enc.shape[3])
        mask[:, : , y : y + better_patch_size_y, x : x + better_patch_size_x] = occluded_value
        mask_enc = crypten.cryptensor(mask, src = 1)

        # Putting the mask on the picture
        occluded_enc = data_enc * mask_enc

        # Calculating the occluded_output
        with torch.no_grad():
            occluded_output = private_model(occluded_enc)

        # Calculating the absolute difference of the original output and the occluded output
        diff_enc = (original_output - occluded_output).abs().sum()

        # Putting the calculated difference on the encrypted heatmap
        # By creating a patch that has value one on the relevant region
        # Only where the value is one the diff will be added to the heatmap
        place_heatmap = torch.zeros(heatmap.shape[0], heatmap.shape[1])
        place_heatmap[y : y  + better_patch_size_y, x : x + better_patch_size_x] = 1
        heatmap = heatmap + diff_enc * crypten.cryptensor(place_heatmap, src = 1)

        # Calculates the difference but in the range 0 to 1
        comp = ((original_output.softmax(1) - occluded_output.softmax(1)).abs().sum())/ 2

        # Decision if the patch should be divided
        # Difference needs to be bigger than threshold 
        # and patch must be bigger than smallest patch
        # Decision if patch is bigger than the smallest patch needs to be encrypted
        differentiate = (comp > threshold_enc) * crypten.cryptensor((float(current_patch_size)/2.0 >= float(minimal_patch)))
        
        # If decision is true, divide the patch 
        if differentiate.get_plain_text().item(): # Needed to be decrypted due to too complicated logic for crypten.where
            half_patch = current_patch_size // 2
            # Calculate possible positions and set new positions
            for poss_y in [0, half_patch]:
                for poss_x in [0, half_patch]:
                    new_y = poss_y + y
                    new_x = poss_x + x
                    # Check if patch position is bigger than picture
                    # if not start recursion
                    if (new_y < data_enc.shape[2] and new_x < data_enc.shape[3]):
                        heatmap = patch_partition((new_y, new_x), half_patch, heatmap)
        
        crypten.print(f"\rNumber of iterations: {num_it}", end='', flush=True)
        num_it += 1
        
        return heatmap
    
    # Start of occlusion analysis
    for y in range(0, data_enc.shape[2], initial_patch):
        for x in range(0, data_enc.shape[3], initial_patch):
            heatmap_enc = patch_partition((y,x), initial_patch, heatmap_enc)          
    
    # Decrypting the heatmap         
    heatmap = heatmap_enc.get_plain_text()

    # Normalizing the heatmap
    heatmap_norm = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min())

    # Saving the heatmap as a picture
    plt.imshow(heatmap_norm, cmap='hot')
    plt.savefig('heatmap_ho.png')

def main():
    """ Runs the script """
    onnx_model_bytes = prepare_resnet18()
    process_image("test_image.jpg","/tmp/processed_image.pth")
    run_script(onnx_model_bytes) 
    dist.destroy_process_group() # Group needs to be destroyed in the end

if __name__ == '__main__':
    main()