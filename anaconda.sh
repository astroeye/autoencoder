#아나콘다 버전 확인
conda --version
 
#아나콘다 업데이트
conda update conda

#파이썬 버전 확인
python --version

#아나콘다 가상환경 생성
conda create --name(-n) 가상환경명 설치할패키지
 
# 예) 파이썬 3.8.12 버전 설치 & sogang04 이름으로 가상환경 생성
conda create --name sogang04 python=3.8.12
conda activate sogang04

# tensorflow-datasets 먼저 설치 해야 함
conda install tensorflow-datasets   # 1.2.0 is installed
# python=3.8.12 is installed   
# tensorflow=2.3.0 is installed  
# numpy=1.21.2 is installed
conda install pandas                # 1.3.3 is installed
conda install imageio               # 2.9.0 is installed
conda install matplotlib            # 3.4.2 is installed

# 'sogang04'라는 이름의 가상환경 삭제
C:\Users\dhqlwm8xkdnj0\anaconda3\Scripts\activate
conda activate base
conda remove --name sogang04 --all

#설치 된 가상환경 리스트 확인
conda info --envs
#or
conda env list
 
#가상환경 활성화 
#예) activate test
activate sogang04

# 패키지 설치 전 설치 가능한 버전 확인
conda search tensorflow

#라이브버리 설치
conda install numpy
conda install tensorflow=2.6.0

# 설치된 라이브러리 확인
conda list

#이는 현재 사용하고 있는 가상환경에서 ‘env’ 라는 이름이 포함된 ‘패키지’를 검색하는 명령어
conda list env

#이는 현재 사용하고 있는 가상환경에서 ‘env’ 라는 이름의 '패키지' 삭제
conda remove env

#가상환경 비활성화 
#예) deactivate test
deactivate sunday

# tensorflow-datasets 업다고 나올때
conda install -c anaconda tensorflow-datasets

# pytorch
# https://pytorch.org/
conda create --name sogang04
conda activate sogang04
conda install pytorch torchvision torchaudio cpuonly -c pytorch

# gym
# https://anaconda.org/conda-forge/gym
conda install -c conda-forge gym

# matplotlib
# https://anaconda.org/conda-forge/matplotlib
conda install -c conda-forge matplotlib

# collection
# https://anaconda.org/lightsource2-tag/collection
conda install -c lightsource2-tag collectio