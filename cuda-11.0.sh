
그래픽 드라이버
    다운로드 및 실행하여 설치
        https://www.nvidia.co.kr/download/Find.aspx?lang=kr
        472.98-quadro-rtx-desktop-notebook-win10-win11-64bit-international-whql.exe

CUDA Toolkit 11.2 Update 2 Downloads
    다운로드 및 실행하여 설치
        https://developer.nvidia.com/cuda-toolkit-archive
        https://developer.nvidia.com/cuda-11.2.2-download-archive
        https://developer.download.nvidia.com/compute/cuda/11.2.2/local_installers/cuda_11.2.2_461.33_win10.exe
        cuda_11.2.2_461.33_win10.exe
    확인 1
        https://docs.nvidia.com/cuda/archive/11.2.2/
        https://docs.nvidia.com/cuda/archive/11.2.2/cuda-installation-guide-microsoft-windows/index.html
        cmd창 새로 열고
            c:\> nvcc -V
    확인 2 - deviceQuery로 확인
        C:\ProgramData\NVIDIA Corporation\CUDA Samples\v10.1\1_Utilities\deviceQuery
        컴파일해서 확인 -> 모르겠음

cuDNN 8.1.1.33
    cuDNN Installation guide
        https://docs.nvidia.com/deeplearning/cudnn/archives/index.html
        https://docs.nvidia.com/deeplearning/cudnn/archives/cudnn-811/install-guide/index.html
    다운로드
        https://developer.nvidia.com/rdp/cudnn-archive
        https://developer.nvidia.com/compute/machine-learning/cudnn/secure/8.1.1.33/11.2_20210301/cudnn-11.2-windows-x64-v8.1.1.33.zip
        cudnn-11.2-windows-x64-v8.1.1.33.zip
    압축해제 후 덮어쓰기
        C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.2
    환경변수 설정
        Variable Name: CUDA_PATH 
        Variable Value: C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.2
    # Installing zlib # not for 8.1.1.33
    #     https://docs.nvidia.com/deeplearning/cudnn/archives/cudnn-832/install-guide/index.html#install-zlib-windows


pip를 사용하여 TensorFlow 설치
    https://www.tensorflow.org/install/pip

    Python 3.6–3.9
        Python 3.9 지원에는 TensorFlow 2.5 이상이 필요합니다.
        Python 3.8 지원에는 TensorFlow 2.2 이상이 필요합니다
    
    Windows 7 이상(64비트)
        Visual Studio 2015, 2017 및 2019용 Microsoft Visual C++ 재배포 가능 패키지
            https://docs.microsoft.com/ko-kr/cpp/windows/latest-supported-vc-redist?view=msvc-170
            https://aka.ms/vs/17/release/vc_redist.x64.exe
            vc_redist.x64.exe
    
    Windows에서 긴 경로가 사용 설정되었는지 확인합니다.
        https://superuser.com/questions/1119883/windows-10-enable-ntfs-long-paths-policy-option-missing
        python 설치할때 긴 경로 관련 설정 나옴
    

python코드 GPU 실행
    https://www.tensorflow.org/install/pip
        Python 3.6~3.9, pip 19.0 이상이 필요합니다.
            python-3.9.10 설치
                https://www.python.org/downloads/release/python-3910/
                python-3.9.10-amd64.exe
            가상환경 만들기
                C:\workspace\autoencoder>python -m venv --system-site-packages .\venv
            가상환경 활성화
                C:\workspace\autoencoder>.\venv\Scripts\activate
            호스트 시스템 설정에 영향을 주지 않고 가상 환경 내에 패키지를 설치합니다.
            pip 업그레이드로 시작합니다.
                (venv) C:\workspace\autoencoder>pip install --upgrade pip
                (venv) C:\workspace\autoencoder>pip list
            가상 환경을 나중에 종료하려면 다음 단계를 따르세요.
                (venv) C:\workspace\autoencoder>deactivate
        TensorFlow pip 패키지 설치
            (venv) C:\workspace\autoencoder>pip install --upgrade tensorflow

        나머지 패키지 설치
        https://pypi.org/
            (venv) C:\workspace\autoencoder>pip install numpy
            (venv) C:\workspace\autoencoder>pip install matplotlib
            (venv) C:\workspace\autoencoder>pip install pandas
            (venv) C:\workspace\autoencoder>pip install scikit-learn
            (venv) C:\workspace\autoencoder>pip install autokeras

        그래픽 드라이버 472.98
        CUDA Toolkit 11.2 Update 2
        cuDNN 8.1.1.33
        python-3.9.10

        autokeras                    1.0.18
        tensorflow                   2.8.0
        matplotlib                   3.5.1
        numpy                        1.22.3
        pandas                       1.4.1
        scikit-learn                 1.0.2

######################################################################################

        python=3.8.12
        tensorflow-gpu=2.6.0

# C:\Users\astro\anaconda3\envs\ae
# C:\Users\dhqlwm8xkdnj0\anaconda3\Scripts\activate
conda activate base
conda remove --name tf-gpu-uda11.0 --all -y

# conda create --name ae python=3.8.12
conda create --name ae python=3.9.7
# conda install -c anaconda python=3.9.7
conda activate ae
conda install -c anaconda tensorflow-gpu==2.6.0

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