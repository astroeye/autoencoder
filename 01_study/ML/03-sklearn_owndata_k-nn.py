# https://wooono.tistory.com/83?category=918936
# [ML] 생성한 데이터셋으로 다중클래스 분류 (K-NN 모델)

# csv 데이터를 읽어와 DataFrame 생성

import pandas as pd

df = pd.read_csv('./spmm dataset 1024 2048 4096 8192.csv')
df

# DataFrame에서 data와 label을 분할해 numpy 배열로 생성
data = df.values[:,1:4]
label = df.values[:,4]

# label에 따른 target_names 지정
import numpy as np
target_names = np.array(['spark sm*dm','breeze sm*sm','breeze sm*dm','breeze dm*sm'])

# 데이터셋을 훈련세트와 테스트세트로 나눈다.

#  scikit-learn은 데이터셋을 섞어서 나눠주는 train_test_split 함수를 제공합니다.
#   이 함수는 전체 행 중 75%를 훈련 세트로, 나머지 25%는 테스트 세트로 만들어줍니다.
#   scikit-learn에서 데이터는 대문자 X로 표시하고 레이블은 소문자 y로 표기합니다.

#  train_test_split 함수로 데이터를 나누기 전에 유사 난수 생성기를 사용해 데이터셋을 무작위로 섞어야 합니다.
#   이 함수를 여러 번 실행해도 결과가 똑같이 나오도록 유사 난수 생성기에 넣을 난수 초깃값을 random_state 매개변수로 전달합니다.
#   이렇게 하면 이 코드는 항상 같은 결과를 출력합니다.

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
data,label,random_state=0
)
# train, test 데이터셋 출력
#  1차원 shape : (숫자, )
#  2차원 shape : (숫자, 숫자)

print("X_train 크기: {}".format(X_train.shape))
print("y_train 크기: {}".format(y_train.shape))
print("X_test 크기: {}".format(X_test.shape))
print("y_test 크기: {}".format(y_test.shape))
# X_train 크기: (653, 3)
# y_train 크기: (653,)
# X_test 크기: (218, 3)
# y_test 크기: (218,)

# 모델 생성
#  skikit-learn에서 제공하는 K-최근접 이웃 모델 사용
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=1)

#  훈련 데이터셋으로부터 모델 생성
#    knn 객체의 fit 메서드를 사용

knn.fit(X_train, y_train)
# 훈련된 모델에 샘플값(2차원 Numpy)을 넣어 class 예측해보기
#  knn 객체의 predict 메서드를 사용

X_new = np.array([[2048, 0.01, 0.03]])

prediction = knn.predict(X_new)

print("예측: {}".format(prediction))
print("예측한 타깃의 이름: {}".format(target_names[int(prediction)]))
# 예측: [1.]
# 예측한 타깃의 이름: breeze sm*sm

# 모델 평가하기
#  테스트 세트를 사용해 정확도를 계산하여 모델의 성능을 평가합니다.

y_pred = knn.predict(X_test)

print("테스트 세트에 대한 예측값:\n {}".format(y_pred))
print("테스트 세트의 정확도: {:.2f}".format(np.mean(y_pred == y_test)))
# 테스트 세트에 대한 예측값:
# [0. 3. 1. 1. 3. 1. 1. 3. 1. 1. 0. 1. 1. 0. 0. 0. 1. 3. 3. 0. 1. 3. 0. 3.
# 1. 3. 1. 0. 0. 0. 3. 1. 0. 0. 1. 1. 1. 3. 1. 1. 0. 0. 0. 3. 0. 0. 0. 1.
# 3. 1. 3. 1. 1. 0. 0. 3. 1. 1. 1. 1. 1. 1. 0. 3. 0. 1. 3. 1. 0. 3. 0. 1.
# 1. 3. 0. 0. 1. 1. 0. 1. 0. 1. 0. 0. 0. 1. 0. 1. 1. 1. 3. 0. 3. 3. 1. 1.
# 3. 1. 3. 0. 1. 3. 1. 1. 3. 0. 1. 1. 0. 0. 3. 0. 1. 1. 1. 3. 2. 1. 3. 0.
# 3. 1. 3. 3. 0. 1. 1. 0. 1. 0. 3. 0. 0. 1. 0. 3. 0. 0. 3. 0. 0. 3. 3. 0.
# 1. 0. 1. 1. 0. 3. 0. 1. 1. 3. 0. 3. 0. 3. 0. 3. 3. 1. 0. 1. 0. 1. 0. 0.
# 3. 0. 0. 0. 1. 1. 0. 1. 3. 0. 3. 1. 0. 0. 0. 0. 1. 0. 3. 3. 3. 1. 0. 0.
# 0. 0. 0. 0. 0. 0. 1. 2. 1. 0. 1. 3. 1. 1. 3. 0. 1. 1. 3. 1. 1. 1. 1. 0.
# 3. 3.]

# 테스트 세트의 정확도: 0.89