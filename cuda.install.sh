
1. 크래픽 카드 별 다운받아야 하는 버전 확인
 - https://en.wikipedia.org/wiki/CUDA#Version_features_and_specifications
 - 그래픽카드: NVIDIA RTX A2000 Laptop GPU 의 컴퓨팅 능력은?
 - 그래픽카드 호환성: compute capability 8.6
 - compute capability 8.6은 CUDA 버전 몇을 지원할까?
 - 호환성별 CUDA 지원 버전: CUDA SDK 11.1 - 11.6 support for compute capability 3.5 - 8.6

 - CUDA 11.1 ~ 11.6를 지원하는 tensorflow_gpu, cuDNN, python 버전을 확인해보자
 	https://www.tensorflow.org/install/source_windows?hl=ko#gpu
	
	GPU 버전	:	tensorflow_gpu-2.5.0
	Python	:	3.6~3.9
	컴파일러	:	MSVC 2019
	빌드도구	:	Bazel 3.7.2
	cuDNN	:	8.1
	CUDA	:	11.2
	
	GPU 버전	:	tensorflow_gpu-2.6.0
	Python	:	3.6~3.9
	컴파일러	:	MSVC 2019
	빌드도구	:	Bazel 3.7.2
	cuDNN	:	8.1
	CUDA	:	11.2
	
	GPU 버전	:	tensorflow_gpu-2.7.0
	Python	:	3.7~3.9
	컴파일러	:	MSVC 2019
	빌드도구	:	Bazel 3.7.2
	cuDNN	:	8.1
	CUDA	:	11.2

2. 버전 정리
	GPU 버전	:	tensorflow_gpu-2.7.0
	Python	:	3.8.12
	컴파일러	:	MSVC 2019
	빌드도구	:	Bazel 3.7.2
	cuDNN	:	8.1.1
	CUDA	:	11.2.2

그래픽 드라이버
472.98-quadro-rtx-desktop-notebook-win10-win11-64bit-international-dch-whql.exe
https://www.nvidia.co.kr/download/Find.aspx?lang=kr

그래픽 드라이버가 지원하는 CUDA 버전 확인
c:\> nvidia-smi

CUDA Toolkit 11.2.2 설치
cuda_11.2.2_461.33_win10.exe
https://developer.nvidia.com/cuda-toolkit-archive





Installing zlib 압축풀어 환경변수 설정
zlib123dllx64.zip
https://docs.nvidia.com/deeplearning/cudnn/install-guide/index.html#install-zlib-windows
저장하고 환경변수 설정: C:\nvidia_cudnn\zlib123dllx64\dll_x64 

cuDNN 8.1.1 파일 덮어쓰기
https://developer.nvidia.com/rdp/cudnn-archive
C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.2 에 덮어쓰기

환경변수 설정
CUDA_PATH
C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.2

c:\>nvcc --version
c:\>python
>>> import tensorflow as tf
>>> print(tf.__version__) # Tensorflow 버전 확인
2.7
>>> tf.test.is_built_with_cuda() # CUDA로 빌드되는지 확인
True
>>> tf.test.is_built_with_gpu_support() # CUDA와 같은 GPU로 빌드되는지 확인
True
>>> tf.test.gpu_device_name() # 사용가능한 GPU 기기들 출력


2. 버전 정리
	GPU 버전	:	tensorflow_gpu-2.6.0
	GPU 버전	:	tensorflow_gpu-2.7.0
	Python	:	3.8.12
	컴파일러	:	MSVC 2019
	빌드도구	:	Bazel 3.7.2
	cuDNN	:	8.1.1
	CUDA	:	11.2.2
	
3. Anaconda python, tensorflow, tensorflow_gpu 설치
conda activate base
conda remove --name tf --all

conda create --name tf python=3.8.12
conda activate ae
conda install -c anaconda tensorflow-datasets

conda install pandas                # 1.3.3 is installed
conda install imageio               # 2.9.0 is installed
conda install matplotlib            # 3.4.2 is installed
conda install -c anaconda scikit-learn


# GPU 인식 여부 확인
from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())


(tf) C:\workspace\autoencoder>python
Python 3.8.12 | packaged by conda-forge | (default, Oct 12 2021, 21:22:46) [MSC v.1916 64 bit (AMD64)] on win32
Type "help", "copyright", "credits" or "license" for more information.
>>> from tensorflow.python.client import device_lib
>>> print(device_lib.list_local_devices())
2022-02-13 09:42:03.823925: I tensorflow/core/platform/cpu_feature_guard.cc:151] This TensorFlow binary is optimized with oneAPI Deep Neural Network Library (oneDNN) to use the following CPU instructions in performance-critical operations:  AVX AVX2
To enable them in other operations, rebuild TensorFlow with the appropriate compiler flags.
2022-02-13 09:42:04.237386: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1525] Created device /device:GPU:0 with 1664 MB memory:  -> device: 0, name: NVIDIA RTX A2000 Laptop GPU, pci bus id: 0000:01:00.0, compute capability: 8.6
[name: "/device:CPU:0"
device_type: "CPU"
memory_limit: 268435456
}
incarnation: 2849803001405483193
xla_global_id: -1
, name: "/device:GPU:0"
device_type: "GPU"
memory_limit: 1745276110
locality {
  bus_id: 1
  links {
  }
}
incarnation: 14464565470727985510
physical_device_desc: "device: 0, name: NVIDIA RTX A2000 Laptop GPU, pci bus id: 0000:01:00.0, compute capability: 8.6"
xla_global_id: 416903419
]

# CPU 학습
print("CPU를 사용한 학습")
with tf.device("/device:CPU:0"):
    model.fit(x_train, y_train, batch_size=32, epochs=3)

print("GPU를 사용한 학습")
with tf.device("/device:GPU:0"):
    model.fit(x_train, y_train, batch_size=32, epochs=3)

	
conda search tensorflow		# recently version is 2.6.0
conda search tensorflow-gpu	# recently version is 2.6.0
conda list tensorflow

pip freeze | findstr "tensorflow"
pip install autokeras
conda install -c anaconda tensorflow-gpu=2.6.0


