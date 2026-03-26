# 2 Computer setup
## Prerequisites
- 2 computers that can reach each other
- CrypTen is installed
- Files to be executed have been copied to the Implementation folder.

## Included files/folders
- script_dec_sw_a: Unencrypted sliding window algorithm from Alice-Net
- script_dec_ho_a: Unencrypted hierarchical algorithm from Alice-Net
- script_dec_mc_a: Unencrypted Monte-Carlo algorithm from Alice-Net
- script_enc_sw_a: Encrypted sliding window algorithm from Alice-Net
- script_enc_ho_a: Encrypted hierarchical algorithm from Alice-Net
- script_enc_mc_a: Encrypted Monte Carlo algorithm from Alice-Net
- script_dec_sw_r: Unencrypted sliding window algorithm from ResNet18
- script_dec_ho_r: Unencrypted hierarchical algorithm from ResNet18
- script_dec_mc_r: Unencrypted Monte Carlo algorithm from ResNet18
- script_enc_sw_r: Encrypted sliding window algorithm from ResNet18
- script_enc_ho_r: Encrypted hierarchical algorithm from ResNet18
- script_enc_mc_r: Encrypted Monte Carlo algorithm from ResNet18 
- Runtime-Test: Folder containing the runtime tests in Alice-Net's two-computer setup (the files have the same names as listed here)

## Usage
All files in Implementation must be available on both computers. The computer that is to load the model should be executed as rank 0. Its address is also used as the master address. The other computer provides the data. If you are working with Alice-Net, you must first execute 
`python mnist_utils.py --option train_v_test` in the terminal of the data owner. Now start the terminal (sourced in Conda) on both computers. For the model owner, execute the code with
`export GLOO_LOG_LEVEL=DEBUG 
export NCCL_DEBUG=INFO
export GLOO_SOCKET_IFNAME="enp0s3"
export WORLD_SIZE=2
export RENDEZVOUS=tcp://0.0.0.0:29500
export RANK=0
python <script>.py --master_addr 0.0.0.0:29500 --rank 0 --world_size 2` and with the data owner with 
`export GLOO_LOG_LEVEL=DEBUG
export NCCL_DEBUG=INFO
export GLOO_SOCKET_IFNAME="enp0s3"
export WORLD_SIZE=2
export RENDEZVOUS=tcp://<IP of the other computer>:29500
export RANK=0
python <script>.py --master_addr <IP of the other computer>:29500 --rank 1 --world_size 2`. 
For unencrypted files, the information about IP, rank, and worldsize must still be entered manually in the code.

In the runtime tests of the unencrypted setting at Alice-Net, the time for each iteration of the occlusion analysis must be added to the measurement so that Bob sends the image with the mask and Alice receives it. In the paper to this code, this time was approximately 9 ms.
