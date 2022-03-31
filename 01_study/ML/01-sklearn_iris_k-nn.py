# https://wooono.tistory.com/81?category=918936
# [ML] Iris 데이터셋을 사용한 다중클래스 분류 (K-NN 모델)

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

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
iris_dataset['data'], iris_dataset['target'], random_state=0)

# train_test_split 함수로 데이터를 나누기 전에 유사 난수 생성기를 사용해 데이터셋을 무작위로
#  섞어야 합니다. 만약 섞지 않은 상태로 데이터셋 뒤쪽의 25%를 테스트 세트로 이용한다면
#  그 데이터의 레이블은 모두 2가 됩니다. 데이터 포인트가 레이블 순서대로 정렬되어 있기 때문입니다
#  (앞에서 iris_dataset[‘target’]의 출력을 확인하세요).
#  세 클래스 중 하나만 포함한 테스트 세트를 사용하면 모델이 얼마나 잘 일반화되었는지 알 수 없습니다.
#  따라서 테스트 세트가 세 클래스의 데이터를 모두 포함하도록 섞어야 합니다.

# 이 함수를 여러 번 실행해도 결과가 똑같이 나오도록 유사 난수 생성기에 넣을 난수 초깃값을
#  random_state 매개변수로 전달합니다. 이렇게 하면 이 코드는 항상 같은 결과를 출력합니다.
#  이 책에서 난수를 사용할 때는 항상 이렇게 random_state를 고정할 것입니다.

# train_test_split 함수의 반환값은 X_train, X_test, y_train, y_test이며 모두 NumPy 배열입니다.
#  X_train은 전체 데이터셋의 75%를, X_test는 나머지 25%를 담고 있습니다.
print("X_train 크기: {}".format(X_train.shape))
print("y_train 크기: {}".format(y_train.shape))
# X_train 크기: (112, 4)
# y_train 크기: (112,)

print("X_test 크기: {}".format(X_test.shape))
print("y_test 크기: {}".format(y_test.shape))
# X_test 크기: (38, 4)
# y_test 크기: (38,)

# [가장 먼저 할 일: 데이터 살펴보기]

# 머신러닝 모델을 만들기 전에 머신러닝이 없이도 풀 수 있는 문제는 아닌지,
#  혹은 필요한 정보가 누락되지는 않았는지 데이터를 조사해보는 것이 좋습니다.

# 또한 데이터를 탐색하면서 비정상적인 값이나 특이한 값들을 찾을 수도 있습니다.
#  예를 들어 붓꽃 데이터 중 일부는 센티미터가 아니고 인치로 되어 있을 수도 있습니다.
#  실제로 데이터에 일관성이 없거나 이상한 값이 들어가 있는 경우가 종종 있습니다.

# 시각화는 데이터를 조사하는 아주 좋은 방법입니다. 산점도scatter plot가 그중 하나입니다.
#  산점도는 데이터에서 한 특성을 x 축에 놓고 다른 하나는 y 축에 놓아 각 데이터 포인트를
#  하나의 점으로 나타내는 그래프입니다. 아쉽게도 컴퓨터 화면은 2차원이라 한 번에
#  2개의 특성만 그릴 수 있습니다. 따라서 이 그래프로는 3개 이상의 특성을 표현하기 어렵습니다.
#  대신 모든 특성을 짝지어 만드는 산점도 행렬scatter matrix을 사용할 수 있습니다.
#  4개의 특성을 가진 붓꽃의 경우처럼 특성의 수가 적다면 꽤 괜찮은 방법입니다.
#  하지만 잊지 말아야 할 것은, 산점도 행렬은 한 그래프에 모든 특성의 관계가 나타나는 것이
#  아니기 때문에 각각의 나누어진 산점도 그래프에는 드러나지 않는 중요한 성질이 있을 수 있습니다.

# 그림은 훈련 세트를 사용해 만든 4개 특성에 대한 산점도 행렬입니다.
#  데이터 포인트의 색은 붓꽃의 품종에 따라 구분했습니다.
#  이 그래프를 그리려면 먼저 NumPy 배열을 pandas의 DataFrame으로 변경해야 합니다.
#  pandas는 산점도 행렬을 그려주는 scatter_matrix 함수를 제공합니다.

# X_train 데이터를 사용해서 데이터프레임을 만듭니다.
# 열의 이름은 iris_dataset.feature_names에 있는 문자열을 사용합니다.
iris_dataframe = pd.DataFrame(X_train, columns=iris_dataset.feature_names)
# 데이터프레임을 사용해 y_train에 따라 색으로 구분된 산점도 행렬을 만듭니다.
pd.plotting.scatter_matrix(iris_dataframe, c=y_train, figsize=(15, 15), marker='o',
hist_kwds={'bins': 20}, s=60, alpha=.8, cmap=mglearn.cm3)

# 클래스 레이블을 색으로 구분한 iris 데이터셋의 산점도 행렬
# 그래프를 보면 세 클래스가 꽃잎과 꽃받침의 측정값에 따라 비교적 잘 구분되는 것을
#  알 수 있습니다. 이것으로 미루어보아 클래스를 잘 구분하도록 머신러닝 모델을
#  학습시킬 수 있을 것입니다.
# 이 그림의 대각선에 위치한 그래프는 각 특성의 히스토그램입니다.

# [첫 번째 머신러닝 모델: k-최근접 이웃 알고리즘]
# 이제 실제 머신러닝 모델을 만들어보겠습니다. scikit-learn은 다양한 분류 알고리즘을 제공합니다.
#  여기서는 비교적 이해하기 쉬운 k-최근접 이웃k-Nearest Neighbors, k-NN 분류기를 사용하겠습니다.
#  이 모델은 단순히 훈련 데이터를 저장하여 만들어집니다. 새로운 데이터 포인트에 대한 예측이
#  필요하면 알고리즘은 새 데이터 포인트에서 가장 가까운 훈련 데이터 포인트를 찾습니다.
#  그런 다음 찾은 훈련 데이터의 레이블을 새 데이터 포인트의 레이블로 지정합니다.

# k-최근접 이웃 알고리즘에서 k는 가장 가까운 이웃 ‘하나’가 아니라 훈련 데이터에서 새로운
#  데이터 포인트에 가장 가까운 ‘k개’의 이웃을 찾는다는 뜻입니다(예를 들면 가장 가까운
#  세 개 혹은 다섯 개의 이웃). 그런 다음 이 이웃들의 클래스 중 빈도가 가장 높은 클래스를
#  예측값으로 사용합니다. 자세한 내용은 2장에서 보도록 하고, 지금은 하나의 이웃만 사용하겠습니다.

# scikit-learn의 모든 머신러닝 모델은 Estimator라는 파이썬 클래스로 각각 구현되어 있습니다.
#  k-최근접 이웃 분류 알고리즘은 neighbors 모듈 아래 KNeighborsClassifier 클래스에
#  구현되어 있습니다. 모델을 사용하려면 클래스로부터 객체를 만들어야 합니다.
#  이때 모델에 필요한 매개변수를 넣습니다.
#  KNeighborsClassifier에서 가장 중요한 매개변수는 이웃의 개수입니다. 우리는 1로 지정하겠습니다.

# knn 객체는 훈련 데이터로 모델을 만들고 새로운 데이터 포인트에 대해 예측하는 알고리즘을
#  캡슐화한 것입니다. 또한 알고리즘이 훈련 데이터로부터 추출한 정보를 담고 있습니다.

# KNeighborsClassifier의 경우는 훈련 데이터 자체를 저장하고 있습니다.

from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=1)

# 훈련 데이터셋으로부터 모델을 만들려면 knn 객체의 fit 메서드를 사용합니다.

# 이 메서드는 훈련 데이터인 NumPy 배열 X_train과 훈련 데이터의 레이블을 담고 있는
#  NumPy 배열 y_train을 매개변수로 받습니다.

knn.fit(X_train, y_train)
# KNeighborsClassifier(n_neighbors=1)

# fit 메서드는 knn 객체 자체를 반환합니다(그리고 knn 객체 자체를 변경시킵니다).
#  그래서 knn 객체가 문자열 형태로 출력됩니다. 이 출력에서 모델을 생성할 때 사용한 매개변수를
#  볼 수 있습니다. 거의 모든 매개변수가 기본값이고 n_neighbors=1은 우리가 지정한 값입니다.
#  scikit-learn 모델들이 많은 매개변수를 가지고 있지만 대부분은 성능을 최적화하거나 특별한
#  목적으로 사용합니다. 앞의 출력 결과에 있는 이런 매개변수들을 신경 쓸 필요는 없습니다.

# scikit-learn 모델을 출력하면 매우 긴 내용이 나올 수 있지만 겁먹지 않아도 됩니다.
#  2장에서 중요 매개변수를 모두 다루겠습니다. fit 메서드의 출력에 어떤 새로운 정보가 있는 것은
#  아니므로 이 책의 나머지 부분에서는 나타내지 않겠습니다.

# [예측하기]
# 이제 이 모델을 사용해서 정확한 레이블을 모르는 새 데이터에 대해 예측을 만들 수 있습니다.
#  야생에서 꽃받침의 길이가 5cm, 폭이 2.9cm이고 꽃잎의 길이가 1cm, 폭이 0.2cm인 붓꽃을 보았다고
#  가정합시다. 그럼 이 붓꽃의 품종은 무엇일까요? 먼저 이 측정값을 NumPy 배열, 즉 샘플의 수(1)에
#  특성의 수(4)를 곱한 크기의 NumPy 배열로 만들어보겠습니다.

X_new = np.array([[5, 2.9, 1, 0.2]])
print("X_new.shape: {}".format(X_new.shape))
# X_new.shape: (1, 4)

# 붓꽃 하나의 측정값은 2차원 NumPy 배열에 행으로 들어갑니다.
#  scikit-learn은 항상 데이터가 2차원 배열일 것으로 예상합니다.

# 예측에는 knn 객체의 predict 메서드를 사용합니다.

prediction = knn.predict(X_new)
print("예측: {}".format(prediction))
print("예측한 타깃의 이름: {}".format(
iris_dataset['target_names'][prediction]))
# 예측: [0]
# 예측한 타깃의 이름: ['setosa']

# 우리가 만든 모델이 새로운 붓꽃을 setosa 품종을 의미하는 클래스 0으로 예측했습니다.
#  그런데 어떻게 이 모델의 결과를 신뢰할 수 있을까요? 이 샘플의 정확한 품종을 모른다는
#  사실이 모델을 구축하는 데 있어서 중요한 의미를 가집니다.

# [모델 평가하기]
# 앞서 만든 테스트 세트를 사용할 때가 왔습니다. 이 데이터는 모델을 만들 때 사용하지 않았고
#  테스트 세트에 있는 각 붓꽃의 품종을 정확히 알고 있습니다.

# 따라서 테스트 데이터에 있는 붓꽃의 품종을 예측하고 실제 레이블(품종)과 비교할 수 있습니다.
#  얼마나 많은 붓꽃 품종이 정확히 맞았는지 정확도를 계산하여 모델의 성능을 평가합니다

y_pred = knn.predict(X_test)
print("테스트 세트에 대한 예측값:\n {}".format(y_pred))
# 테스트 세트에 대한 예측값:
#  [2 1 0 2 0 2 0 1 1 1 2 1 1 1 1 0 1 1 0 0 2 1 0 0 2 0 0 1 1 0 2 1 0 2 2 1 0 2]
print("예측한 타깃의 이름: {}".format(
iris_dataset['target_names'][y_pred]))
# 예측한 타깃의 이름: ['virginica' 'versicolor' 'setosa' 'virginica' 'setosa' 'virginica'
#  'setosa' 'versicolor' 'versicolor' 'versicolor' 'virginica' 'versicolor'
#  'versicolor' 'versicolor' 'versicolor' 'setosa' 'versicolor' 'versicolor'
#  'setosa' 'setosa' 'virginica' 'versicolor' 'setosa' 'setosa' 'virginica'
#  'setosa' 'setosa' 'versicolor' 'versicolor' 'setosa' 'virginica'
#  'versicolor' 'setosa' 'virginica' 'virginica' 'versicolor' 'setosa'
#  'virginica']

print("테스트 세트의 정확도: {:.2f}".format(np.mean(y_pred == y_test)))
# 테스트 세트의 정확도: 0.97

# 이 모델의 테스트 세트에 대한 정확도는 약 0.97입니다. 이 말은 테스트 세트에 포함된 붓꽃 중
#  97%의 품종을 정확히 맞혔다는 뜻입니다.
#  이 결과 이 모델은 새로운 붓꽃에 대한 정확도가 97%일 것이라 기대할 수 있습니다.
#  정확도가 높으므로 아마추어 식물학자는 이 애플리케이션을 충분히 신뢰하고 사용할만합니다.
#  이후의 장들에서 모델의 성능을 높이는 방법과 모델을 튜닝할 때 주의할 점을 살펴보겠습니다