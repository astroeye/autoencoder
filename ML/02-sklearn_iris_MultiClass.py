# [ML] Iris 데이터셋을 사용한 다중클래스 분류 (다층퍼셉트론 모델)

# [Iris 붓꽃의 품종 분류]
# 다중 클래스 분류 대표 예제
# 붓꽃의 4가지 특성 데이터값을 사용해 3가지 붓꽃 품종 중 하나를 예측하는 모델 만들기
# 모델은 K-최근접이웃 사용

# [데이터 적재]
# 우리가 사용할 데이터셋은 머신러닝과 통계 분야에서 오래전부터 사용해온 붓꽃 iris 데이터셋입니다.

# 이 데이터는 scikit-learn의 datasets 모듈에 포함되어 있습니다. load_iris 함수를 사용해서 데이터를 적재하겠습니다.
from sklearn.datasets import load_iris
import numpy as np
import pandas as pd
import mglearn
iris_dataset = load_iris()

# load_iris가 반환한 iris 객체는 파이썬의 딕셔너리 Dictionary 와 유사한 Bunch 클래스의 객체입니다.

# 즉 키와 값으로 구성되어 있습니다.
print("iris_dataset의 키: \n{}".format(iris_dataset.keys()))
# iris_dataset의 키: 
# dict_keys(['data', 'target', 'frame', 'target_names', 'DESCR', 'feature_names', 'filename'])

# DESCR 키에는 데이터셋에 대한 간략한 설명이 들어 있습니다.
print(iris_dataset['DESCR'][:193] + "\n...")
#  Iris plants dataset
# --------------------

# **Data Set Characteristics:**

#     :Number of Instances: 150 (50 in each of three classes)
#     :Number of Attributes: 4 numeric, pre
# ...

# target_names의 값은 우리가 예측하려는 붓꽃 품종의 이름을 문자열 배열로 가지고 있습니다.
print("타깃의 이름: {}".format(iris_dataset['target_names']))
# 타깃의 이름: ['setosa' 'versicolor' 'virginica']

# feature_names의 값은 각 특성을 설명하는 문자열 리스트입니다.
print("특성의 이름: \n{}".format(iris_dataset['feature_names']))
# 특성의 이름: 
# ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']

# 실제 데이터는 target과 data 필드에 들어 있습니다.
# data는 꽃잎의 길이와 폭, 꽃받침의 길이와 폭을 수치 값으로 가지고 있는 NumPy 배열입니다.
print("data의 타입: {}".format(type(iris_dataset['data'])))
# data의 타입: <class 'numpy.ndarray'>

# data 배열의 행은 개개의 꽃이 되며 열은 각 꽃에서 구한 네 개의 측정치입니다.
print("data의 크기: {}".format(iris_dataset['data'].shape))
# data의 크기: (150, 4)
# 이 배열은 150개의 붓꽃 데이터를 가지고 있습니다.
# 머신러닝에서 각 아이템은 샘플이라 하고 속성은 특성이라고 부릅니다.
# 그러므로 data 배열의 크기는 샘플의 수에 특성의 수를 곱한 값이 됩니다.
# 이는 scikit-learn의 스타일이며 항상 데이터가 이런 구조일 거라 가정하고 있습니다.

# 다음은 맨 처음 다섯 샘플의 특성값입니다.
print("data의 처음 다섯 행:\n{}".format(iris_dataset['data'][:5]))
# data의 처음 다섯 행:
# [[5.1 3.5 1.4 0.2]
#  [4.9 3.  1.4 0.2]
#  [4.7 3.2 1.3 0.2]
#  [4.6 3.1 1.5 0.2]
#  [5.  3.6 1.4 0.2]]

# target은 각 원소가 붓꽃 하나에 해당하는 1차원 배열입니다.
print("target의 크기: {}".format(iris_dataset['target'].shape))
# target의 크기: (150,)

# 붓꽃의 종류는 0에서 2까지의 정수로 기록되어 있습니다.
print("타깃:\n{}".format(iris_dataset['target']))
# 타깃:
# [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
#  0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
#  1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 2 2 2 2 2 2 2 2 2 2 2
#  2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2
#  2 2]

# [성과 측정: 훈련 데이터와 테스트 데이터]
# 이 데이터로 머신러닝 모델을 만들고 새로운 데이터의 품종을 예측하려 합니다.

# 하지만 만든 모델을 새 데이터에 적용하기 전에 이 모델이 진짜 잘 작동하는지 알아야 합니다.
#  다시 말해서 우리가 만든 모델의 예측을 신뢰할 수 있는지 알아야 합니다.

# 불행히도 모델을 만들 때 쓴 데이터는 평가 목적으로 사용할 수 없습니다.
#  모델이 훈련 데이터를 그냥 전부 기억할 수 있으니 훈련 데이터에 속한 어떤 데이터라도
#  정확히 맞출 수 있기 때문입니다.
#  이렇게 데이터를 기억한다는 것은 모델을 잘 일반화하지 않았다는 뜻입니다.
#  (다른 말로는 새로운 데이터에 대해서는 잘 작동하지 않는다는 것입니다).

# 모델의 성능을 측정하려면 레이블을 알고 있는 (이전에 본 적 없는) 새 데이터를 모델에 적용해봐야 합니다.
#  이를 위해 우리가 가지고 있는 레이블된 데이터(150개의 붓꽃 데이터)를 두 그룹으로 나눕니다.

# 그중 하나는 머신러닝 모델을 만들 때 사용하며, 훈련 데이터 혹은 훈련 세트training set라고 합니다.
#  나머지는 모델이 얼마나 잘 작동하는지 측정하는 데 사용하며, 이를 테스트 데이터,
#  테스트 세트 test set 혹은 홀드아웃 세트 hold-out set라고 부릅니다.

# scikit-learn은 데이터셋을 섞어서 나눠주는 train_test_split 함수를 제공합니다.
#  이 함수는 전체 행 중 75% 를 레이블 데이터와 함께 훈련 세트로 뽑습니다.
#  나머지 25%는 레이블 데이터와 함께 테스트 세트가 됩니다.
#  훈련 세트와 테스트 세트를 얼만큼씩 나눌지는 상황에 따라 다르지만
#  전체의 25%를 테스트 세트로 사용하는 것은 일반적으로 좋은 선택입니다.

# scikit-learn에서 데이터는 대문자 X로 표시하고 레이블은 소문자 y로 표기합니다.
#  이는 수학에서 함수의 입력을 x, 출력을 y로 나타내는 표준 공식 f(x)=y에서 유래된 것입니다.
#  수학의 표기 방식을 따르되 데이터는 2차원 배열(행렬)이므로 대문자 X를,
#  타깃은 1차원 배열(벡터)이므로 소문자 y를 사용합니다.

# train_test_split의 반환값을 저장하는 데 이 명명 규칙을 사용하겠습니다.

#  클래스별로 확률값을 지정하기 위해서는 label을 “one-hot 인코딩”으로 사용
#   클래스별 확률값의 합은 1.0 이다.

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
iris_dataset['data'], iris_dataset['target'], random_state=0)

from tensorflow.keras.utils import to_categorical

y_train = to_categorical(y_train, num_classes=4) # one-hot 인코딩 [1] -> [0,1,0,0]
y_test = to_categorical(y_test, num_classes=4) # one-hot 인코딩

# train, test 데이터셋 출력
#  1차원 shape : (숫자, )
#  2차원 shape : (숫자, 숫자)

print("X_train 크기: {}".format(X_train.shape))
print("y_train 크기: {}".format(y_train.shape))
print("X_test 크기: {}".format(X_test.shape))
print("y_test 크기: {}".format(y_test.shape))
# X_train 크기: (112, 4)
# y_train 크기: (112, 4)
# X_test 크기: (38, 4)
# y_test 크기: (38, 4)

# 다층 퍼셉트론 신경망 모델 생성
#  활성함수 relu는 오류역전파가 용이하다.
#  활성함수 softmax는 입력되는 값을 클래스별 확률값이 나오도록 출력시킵니다.
#   이 확률 값을 모두 더하면 1이 됩니다.
#   다중 클래스 모델의 출력층에 주로 사용되며, 확률값이 가장 높은 클래스가
#   모델이 분류한 클래스입니다.

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

model = Sequential()

# 입력은 4개 특성값을 가진 1차원 벡터
model.add(Dense(64,input_shape=(4,),activation='relu'))

# 활성함수 relu 
model.add(Dense(64,activation='relu'))

# 출력은 4가지 클래스별 확률값을 가진 1차원 벡터
# 활성함수 softmax
model.add(Dense(4,activation='softmax'))

# 모델 학습 과정 설정
model.compile(loss='categorical_crossentropy', 
optimizer='Adam', 
metrics=['accuracy'])

model.summary()

# 모델 훈련
hist = model.fit(X_train, y_train, epochs=50, batch_size=32)

# 학습과정 살펴보기
#%matplotlib inline
import matplotlib.pyplot as plt

fig, loss_ax = plt.subplots()

acc_ax = loss_ax.twinx()

loss_ax.set_ylim([0.0, 3.0])
acc_ax.set_ylim([0.0, 1.0])

loss_ax.plot(hist.history['loss'], 'y', label='train loss')
acc_ax.plot(hist.history['accuracy'], 'b', label='train acc')

loss_ax.set_xlabel('epoch')
loss_ax.set_ylabel('loss')
acc_ax.set_ylabel('accuray')

loss_ax.legend(loc='upper left')
acc_ax.legend(loc='lower left')

plt.show()

# 모델 평가하기
loss_and_metrics = model.evaluate(X_test, y_test, batch_size=8)
print('loss_and_metrics : ' + str(loss_and_metrics))
# 5/5 [==============================] - 0s 584us/step - loss: 0.5975 - accuracy: 0.9737
# loss_and_metrics : [0.5974655151367188, 0.9736841917037964]