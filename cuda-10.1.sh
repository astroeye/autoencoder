
그래픽 드라이버
    다운로드 및 실행하여 설치
        https://www.nvidia.co.kr/download/Find.aspx?lang=kr
        472.98-quadro-rtx-desktop-notebook-win10-win11-64bit-international-whql.exe

CUDA Toolkit 10.1 update2 Archive
    다운로드 및 실행하여 설치
        https://developer.nvidia.com/cuda-10.1-download-archive-update2
        https://developer.download.nvidia.com/compute/cuda/10.1/Prod/local_installers/cuda_10.1.243_426.00_win10.exe
        cuda_10.1.243_426.00_win10.exe
    확인 1
        https://docs.nvidia.com/cuda/archive/10.1/cuda-installation-guide-microsoft-windows/index.html
        cmd창 새로 열고
            c:\> nvcc -V
    확인 2 - deviceQuery로 확인
        C:\ProgramData\NVIDIA Corporation\CUDA Samples\v10.1\1_Utilities\deviceQuery
        컴파일해서 확인 -> 모르겠음

cuDNN v7.6.5
    다운로드
        https://developer.nvidia.com/rdp/cudnn-archive
        https://developer.nvidia.com/compute/machine-learning/cudnn/secure/7.6.5.32/Production/10.1_20191031/cudnn-10.1-windows10-x64-v7.6.5.32.zip
        cudnn-10.1-windows10-x64-v7.6.5.32.zip
    압축해제 후 덮어쓰기
        C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v10.1

# 참고해보자
https://vitalholic.tistory.com/375

# tensorflow-gpu=2.3.0 설치

conda activate base
conda remove --name ae --all -y

conda create --name ae
conda activate ae
conda install -c anaconda python=3.8.12
pip install tensorflow-gpu==2.3.0

python
>>> import tensorflow as tf
>>> from tensorflow.python.client import device_lib
>>> print(device_lib.list_local_devices())


conda install -c anaconda tensorflow-gpu=2.3.0

conda install -c anaconda tensorflow-datasets   # tensorflow=2.3.0      # tensorflow=datasets-1.2.0
conda install tensorflow-gpu                    # tensorflow-gpu=2.3.0
conda install matplotlib                        # matplotlib=3.4.3
conda install pandas                            # pandas=1.4.1
conda install scikit-learn                      # scikit-learn=1.0.2


# GPU 사용량 확인
for /l %g in () do @(cls & nvidia-smi & timeout /t 1)
