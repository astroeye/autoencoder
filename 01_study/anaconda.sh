$ cat ~/.ssh/id_rsa.pub  #(없으면 아래 실행하여 생성)
$ cd ~/.ssh/
$ ssh-keygen -t rsa
$ cat ~/.ssh/id_rsa.pub # 정보 복사

# Github >> SEttings >> SSH and GPG keys >> SSH keys >> New SSH key
Title: 원하는 title
id_rsa.pub 내용 붙여넣기

$ cd c:\workspace
$ git config --global user.email “astroeye@hotmail.com”
$ git clone git@github.com:astroeye/autoencoder
$ cd autoencoder

# 문서 파일 생성, 수정 또는 삭제 후
$ git add *.*
$ git commit -m "first commit"
$ git branch -M main
$ git remote add origin git@github.com:astroeye/autoencoder.git
$ git push -u origin main

===============================================================
# Git Repository 처음 생성 후 나오는 script
===============================================================
…or create a new repository on the command line
echo "# autoencoder" >> README.md
git init
git add README.md
git commit -m "first commit"
git branch -M main
git remote add origin git@github.com:astroeye/autoencoder.git
git push -u origin main

…or push an existing repository from the command line
git remote add origin git@github.com:astroeye/autoencoder.git
git branch -M main
git push -u origin main
===============================================================


#아나콘다 버전 확인
conda --version
 
#아나콘다 업데이트
conda update conda

#파이썬 버전 확인
python --version

#아나콘다 가상환경 생성
conda create --name(-n) 가상환경명 설치할패키지
 
# 예) 파이썬 3.8.12 버전 설치 & sogang04 이름으로 가상환경 생성
conda create --name ae python=3.8.12 -y
# conda create --name ae python=3.9.7 -y # tensorflow-datasets이 Python 3.9 버전에서는 3.9.0만 지원함
conda activate ae

# tensorflow-datasets 먼저 설치 해야 함
conda install tensorflow-datasets -y
# tensorflow-datasets 1.2.0 is installed
# python=3.8.12 is installed   
# tensorflow=2.3.0 is installed  
# numpy=1.21.2 is installed
conda install -c conda-forge imageio -y
conda install -c anaconda pandas -y
conda install -c conda-forge matplotlib -y
conda install -c anaconda scikit-learn -y
conda install -c conda-forge opencv -y
conda install -c conda-forge tensorflow-hub -y
conda install -c anaconda pillow -y

conda install -c anaconda tensorflow-gpu -y


# 'sogang04'라는 이름의 가상환경 삭제
C:\Users\dhqlwm8xkdnj0\anaconda3\Scripts\activate
conda activate base
conda remove --name ae --all -y

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
conda install -c anaconda scikit-learn

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


'--------------------------------------------'
# GPU 설정 후 conda 가상환경
conda activate base
conda remove --name ae --all -y

conda search -c conda-forge python=3.9  # 3.9.10
conda create --name ae -c conda-forge python=3.9  # 3.9.10 설치 됨
conda activate ae

conda search -c conda-forge tensorflow  # 2.7.0
conda install -c conda-forge tensorflow # 2.6.0 설치 됨

conda search -c conda-forge tensorflow-gpu  # 2.6.0
conda install -c anaconda tensorflow-gpu    # 2.6.0 설치 됨

conda search -c conda-forge matplotlib   # 3.5.1
conda install -c conda-forge matplotlib  # 3.5.1 설치 됨

conda search -c conda-forge pandas  # 1.4.1
conda install -c conda-forge pandas # 1.4.1 설치 됨

conda search -c conda-forge scikit-learn    # 1.0.2
conda install -c conda-forge scikit-learn   # 1.0.2 설치 됨

# GPU 사용량 확인
for /l %g in () do @(cls & nvidia-smi & timeout /t 1)



