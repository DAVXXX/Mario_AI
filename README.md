## PreInstallation

***Need to install the C++ Build Tools:

https://visualstudio.microsoft.com/visual-cpp-build-tools/


You will need conda installed: https://www.anaconda.com/download






## Installation





****Next, create a virtual environment**



conda create --name smbrl python=3.10.12


****Make sure you activate the environment.


conda activate smbrl

***If you get error: CondaError: Run 'conda init' before 'conda activate'

conda init


conda update conda


****Install Cuda (Only for Nvidia GPU's)

conda install nvidia/label/cuda-12.1.0::cuda


****Install pytorch (This will be differnt depending on the Hardware refer to: https://pytorch.org/get-started/locally/ )

conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia


*****Finally, install the rest of the requirements**


pip install -r requirements.txt








****DEBUGGING  &  When completed training******


****Clean up Conda: remove any cached files that might be causing issues.
conda clean --all


****First Run to deactivate the Env:
conda deactivate

****Removes the existing env 
conda env remove --name smbrl
