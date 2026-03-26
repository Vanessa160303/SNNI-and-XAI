# 2 Rechner-Setup
## Voraussetzungen
- 2 Rechner, die sich gegenseitig erreichen können
- CrypTen ist installiert
- Dateien, die ausgeführt werden sollen, sind in den Ordner Implementierung-Ba kopiert worden

## Enthaltene Dateien/Ordner
- script_dec_sw_a: Unverschlüsselter Sliding-Window-Algorithmus von Alice-Net
- script_dec_ho_a: Unverschlüsselter hierarchischer Algorithmus von Alice-Net
- script_dec_mc_a: Unverschlüsselter Monte-Carlo-Algorithmus von Alice-Net
- script_enc_sw_a: Verschlüsselter Sliding-Window-Algorithmus von Alice-Net
- script_enc_ho_a: Verschlüsselter hierarchischer Algorithmus von Alice-Net
- script_enc_mc_a: Verschlüsselter Monte-Carlo-Algorithmus von Alice-Net
- script_dec_sw_r: Unverschlüsselter Sliding-Window-Algorithmus von ResNet18
- script_dec_ho_r: Unverschlüsselter hierarchischer Algorithmus von ResNet18
- script_dec_mc_r: Unverschlüsselter Monte-Carlo-Algorithmus von ResNet18
- script_enc_sw_r: Verschlüsselter Sliding-Window-Algorithmus von ResNet18
- script_enc_ho_r: Verschlüsselter hierarchischer Algorithmus von ResNet18
- script_enc_mc_r: Verschlüsselter Monte-Carlo-Algorithmus von ResNet18
- Laufzeit-Test: Ordner mit den Laufzeittests im 2-Rechner-Setup von Alice-Net (die Dateien besitzen die identischen Namen wie hier aufgeführt)

## Nutzung
Alle Dateien in Implementierung-BA müssen auf beiden Rechnern vorhanden sein. Der Rechner, der das Modell laden soll, soll als Rang 0 ausgeführt werden. Seine Adresse wird auch als die Masteradresse verwendet. Der andere Rechner stellt die Daten zur Verfügung. Wenn mit Alice-Net gearbeitet wird, muss hier noch vorher im Terminal des Datenbesitzers
`python mnist_utils.py --option train_v_test` 
ausgeführt werden. Man startet nun auf beiden Rechnern das Terminal (in Conda gesourced). Bei dem Modellbesitzer führt man den Code mit 
`export GLOO_LOG_LEVEL=DEBUG
export NCCL_DEBUG=INFO
export GLOO_SOCKET_IFNAME="enp0s3"
export WORLD_SIZE=2
export RENDEZVOUS=tcp://0.0.0.0:29500
export RANK=0
python <script>.py --master_addr 0.0.0.0:29500 --rank 0 --world_size 2` aus und bei dem Datenbesitzer mit 
`export GLOO_LOG_LEVEL=DEBUG
export NCCL_DEBUG=INFO
export GLOO_SOCKET_IFNAME="enp0s3"
export WORLD_SIZE=2
export RENDEZVOUS=tcp://<IP des anderen Rechners>:29500
export RANK=0
python <script>.py --master_addr <IP des anderen Rechners>:29500 --rank 1 --world_size 2`. 
Bei den unverschlüsselten Dateien müssen die Infos über IP, Rang und Worldsize noch manuell im Code eingegeben werden.

Bei den Laufzeittests des unverschlüsselten Settings bei Alice-Net muss für die Messung der Zeit jeder Iteration der Occlusion Analysis die Zeit dazugerechnet werden, die Bob benötigt, um das Bild mit Maske zu schicken, und die Alice benötigt, um es zu empfangen. Diese Zeit betrug in dieser Arbeit etwa 9 ms.  

# 2 Computer setup
## Prerequisites
- 2 computers that can reach each other
- CrypTen is installed
- Files to be executed have been copied to the Implementation-Ba folder.

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
- Laufzeit-Test: Folder containing the runtime tests in Alice-Net's two-computer setup (the files have the same names as listed here)

## Usage
All files in Implementierung-BA must be available on both computers. The computer that is to load the model should be executed as rank 0. Its address is also used as the master address. The other computer provides the data. If you are working with Alice-Net, you must first execute 
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
python <script>.py --master_addr <IP des anderen Rechners>:29500 --rank 1 --world_size 2`. 
For unencrypted files, the information about IP, rank, and worldsize must still be entered manually in the code.

In the runtime tests of the unencrypted setting at Alice-Net, the time for each iteration of the occlusion analysis must be added to the measurement so that Bob sends the image with the mask and Alice receives it. In this work, this time was approximately 9 ms.
