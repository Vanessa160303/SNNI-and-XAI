# The combination of secure neural network inference (SNNI) and explainable AI (XAI) 
This project shows the code for secure and explainable inference with neural networks. The method of secure neural network inference is multi-party computation and occlusion analysis was used for explainability.

## Prerequisites
CrypTen must first be installed according to the installation instructions (in a Conda environment).
The installation instructions can be found here: [Github of CrypTen](https://github.com/facebookresearch/CrypTen). The Python version used for the code is 3.12. In addition, Linux was used as the operating system. 

Furthermore, the code was tested directly within the CrypTen folder structure. To do this, the folder Implementation was integrated into the CrypTen directory. The Implementation folder should be put in the parent folder, where the tutorials, benchmark, etc. folders are also located.

## Contents of the Implementation folder:
The Implementation folder contains the following files/folders:
- OSA_AliceNet_Dec: Unencrypted Occlusion Analysis with AliceNet
- OSA_AliceNet_Enc: Encrypted occlusion analysis with AliceNet
- OSA_ResNet18_Dec: Unencrypted occlusion analysis with ResNet18
- OSA_ResNet18_Enc: Encrypted occlusion analysis with ResNet18
- models: contains AliceNet
- mnist_utils.py: required for splitting the MNIST data set
- test_image.jpg: Sample test image for ResNet18

The four files above are Jupyter notebooks and contain the 3 different algorithms of the occlusion analysis based on the networks in the secure and insecure setting. The exact functionality is explained in more detail within the notebooks. They can simply be started via the terminal (which was also sourced with Conda) in the Implementation folder with the command `jupyter notebook`. You can then select the notebook you want to use. The folder model and the file mnist_utils.py were supplied by CrypTen, these do not need to be specifically considered. The ONNX file does not need to be considered specifically either.

- OSA_AliceNet_Dec_Test1
- OSA_AliceNet_Enc_Test1
- OSA_ResNet18_Dec_Test1
- OSA_ResNet18_Enc_Test1

These 4 files contain the tests for the resources, i.e., for the runtime of the algorithms. The encrypted tests also contain the tests for communication between the parties in the MPC. They can simply be started via the terminal (which was also sourced with Conda) in the Implementation folder with the command `jupyter notebook`.

- OSA_AliceNet_Dec_Test2
- OSA_AliceNet_Enc_Test2
- OSA_ResNet18_Dec_Test2 
- OSA_ResNet18_Enc_Test2
- SSIM
- Bilder-Res

These first five files contain the tests for the qualitative evaluation metrics, i.e., sparsity, robustness, and SSIM. They can be easily started via the terminal (which was also sourced with Conda) in the implementation folder with the command `jupyter notebook`. It should be noted that the code for the presets OSA_AliceNet_Enc_Test2 and OSA_ResNet18_Dec_Test2 should be executed first, as this is where the noisy images for testing robustness are created. The Bilder-Res folder also contains the images that were used for the ResNet18 tests. If you want to use these, unzip the individual images into the Implementierung folder.

- 2-Computer-Setup

This folder contains the files for the 2-computer setup. These can also be copied to the Implementation folder and executed. The folder also contains an extra ReadMe file.
