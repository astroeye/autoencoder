

conda create --name ae python=3.8.12
conda activate ae
pip install tensorflow-gpu==2.4.0

python
>>> import tensorflow as tf
>>> from tensorflow.python.client import device_lib
>>> print(device_lib.list_local_devices())

conda install -c anaconda numpy             # 1.21.5
conda install -c conda-forge matplotlib     # 3.5.1
conda install -c anaconda pandas            # 1.1.3
conda install -c anaconda scikit-learn      # 
conda install -c jmcmurray os

conda list tensor
conda list numpy
conda list matplotlib
conda list pandas
conda list scikit-learn

# failed to create cublas handle: CUBLAS_STATUS_NOT_INITIALIZED
https://www.litcoder.com/?p=2509
import os as os
os.environ [ "TF_FORCE_GPU_ALLOW_GROWTH" ] = "true"


pip install autokeras
conda install -c conda-forge tensorflow
conda search -c conda-forge tensorflow
pip install tensorflow==2.4.0