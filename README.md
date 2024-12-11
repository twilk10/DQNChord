# Implementation of DQN model to improve stability proceedures in the Chord protocol
## How to Run
- Create a python vertual environment with conda or pip
- Check your CUDA version on your system:
```
nvidia-smi
```
- Ensure that you have pytorch install from the [pytorch Official Website](https://pytorch.org/get-started/locally/), copy the command according to CUDA version
for example: 
```
conda install pytorch torchvision torchaudio pytorch-cuda=12.4 -c pytorch -c nvidia
```
or 
```
pip install pytorch torchvision torchaudio pytorch-cuda=12.4 -c pytorch -c nvidia
```
- After pytorch is installed, you must install all of the other dependencies via the requirments.txt file
```
pip install -r /path/to/requirements.txt
```
or create your conda environment via the requirements.txt
```
conda create --name <env> --file requirements.txt
```
Additionally you may need to run the following command in order to run the custom environment:
```
pip install -e .
```
run via this command
```
py run_gymnasium_env.py
```
## Quick Info
- Currently there is a trained model Chord_model.pt that you can test. Other than that you can run the training and tweak what ever you would like.