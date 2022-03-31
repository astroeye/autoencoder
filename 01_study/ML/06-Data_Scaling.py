# https://wooono.tistory.com/96?category=918936
# [ML] 데이터 스케일링 (Data Scaling) 이란?

# [스케일링이란?]
# 머신러닝을 위한 데이터셋을 정제할 때, 특성별로 데이터의 스케일이 다르다면 어떤 일이 벌어질까요?
# 예를 들어,
#  X1은 0 부터 1 사이의 값을 갖고
#  X2 는 1000000 부터 1000000000000 사이의 값을 갖고
#  y 는 1000000 부터 100000000 사이의 값을 갖는다고 가정한다면
#  X1 특성은 y 를 예측하는데 큰 영향을 주지 않는 것으로 생각할 수 있습니다.
# 때문에 특성별로 데이터의 스케일이 다르다면, 머신러닝이 잘 동작하지 않을 수 있습니다.
# 따라서, 저희는 데이터 스케일링 작업을 통해, 모든 특성의 범위(또는 분포)를 같게 만들어줘야합니다.

# [주로 사용되는 스케일링 개념]
# Standardization (표준화)
#  특성들의 평균을 0, 분산을 1 로 스케일링하는 것입니다.
#  즉, 특성들을 정규분포로 만드는 것입니다.

#  평균(mean=mu=μ) = 변량의 총합/변량의 개수 = {(계급값)*(도수)}의 총합/도수의 총합
#  편차(deviation) =  (변량)-(평균) = d = y - y̅ , 평균에서 얼마나 떨어져 있느냐, 편차의 합은 0이다
#  분산(variance=Var) = 편차 제곱의 평균 = Σ(편차)²/(변량)의 개수
#  표준편차(standard deviation=sigma,σ) = 분산에 제곱근 씌운것 = √(분산)
#  정규분포(Normal distribution or Gaussian distribution) = N(μ, σ²)
#  표준정규분포(standard normal distribution) = N(0, 1) 평균이 0이고, 표준편차가 1인 정규분포
#  U=(X-μ)/σ
#  X(확률변수)

#  1σ = 34.1%,  ±1σ = 68%
#  2σ = 47.7%,  ±2σ = 95%
#  3σ = 49.86%, ±3σ = 99.7%

# Normalization (정규화)
#  특성들을 특정 범위(주로 [0,1]) 로 스케일링 하는 것입니다.
#  가작 작은 값은 0, 가장 큰 값은 1 로 변환되므로, 모든 특성들은 [0, 1] 범위를 갖게됩니다.

# [scikit-learn 의 scaler 사용 전, 주의 사항]
# 우선, scikit-learn 의 scaler 를 사용하기전에, 주의해야할 점을 먼저 살펴보겠습니다.
# scaler 는 fit 과 transform 메서드를 지니고 있습니다.
# fit 메서드는 훈련 데이터에만 적용해, 훈련 데이터의 분포를 먼저 학습하고
# 그 이후, transform 메서드를 훈련 데이터와 테스트 데이터에 적용해 스케일을 조정해야합니다.
# 따라서, 훈련 데이터에는 fit_transform() 메서드를 적용하고, 테스트 데이터에는 transform() 메서드를 적용해야합니다.
#  fit_transform() 은 fit 과 transform 이 결합된 단축 메서드입니다.
# 또한, 스케일링할 때, 모든 특성의 범위를 유사하게 만드는 것은 중요하지만, 그렇다고 모두 같은 분포로 만들 필요는 없습니다.
#  특성에 따라 각기 다른 스케일링을 적용하는게 유리할 수도 있기 때문입니다.
# 이제 scikit-learn 에서 제공하는 5가지 스케일링 방법을 알아보겠습니다.

# [StandardScaler()]
# 특성들의 평균을 0, 분산을 1 로 스케일링하는 것입니다.
#  즉, 특성들을 정규분포로 만드는 것입니다.
# 최솟값과 최댓값의 크기를 제한하지 않기 때문에, 어떤 알고리즘에서는 문제가 있을 수 있으며
# 이상치에 매우 민감합니다.
# 회귀보다 분류에 유용합니다.

from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
iris_dataset = load_iris()
X_train, X_test, y_train, y_test = train_test_split(
iris_dataset['data'], iris_dataset['target'], random_state=0)

# 변형 객체 생성
std_scaler = StandardScaler()

# 훈련데이터의 모수 분포 저장
std_scaler.fit(X_train)

# 훈련 데이터 스케일링
X_train_scaled = std_scaler.transform(X_train)

# 테스트 데이터의 스케일링
X_test_scaled = std_scaler.transform(X_test)

# 스케일링 된 결과 값으로 본래 값을 구할 수도 있다.
# X_origin = std_scaler.inverse_transform(X_train_scaled)

# [MinMaxScaler()]
# Min-Max Normalization 이라고도 불리며,
#  특성들을 특정 범위(주로 [0,1]) 로 스케일링 하는 것입니다.
# 가작 작은 값은 0, 가장 큰 값은 1 로 변환되므로, 모든 특성들은 [0, 1] 범위를 갖게됩니다.
# 이상치에 매우 민감합니다.
# 분류보다 회귀에 유용합니다.


from sklearn.preprocessing import MinMaxScaler

from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
iris_dataset = load_iris()
X_train, X_test, y_train, y_test = train_test_split(
iris_dataset['data'], iris_dataset['target'], random_state=0)

# 변형 객체 생성
minmax_scaler = MinMaxScaler()

# 훈련데이터의 모수 분포 저장
minmax_scaler.fit(X_train)

# 훈련 데이터 스케일링
X_train_scaled = minmax_scaler.transform(X_train)

# 테스트 데이터의 스케일링
X_test_scaled = minmax_scaler.transform(X_test)

# 스케일링 된 결과 값으로 본래 값을 구할 수도 있다.
# X_origin = minmax_scaler.inverse_transform(X_train_scaled)

# [MaxAbsScaler()]
# 각 특성의 절대값이 0 과 1 사이가 되도록 스케일링합니다.
# 즉, 모든 값은 -1 과 1 사이로 표현되며, 데이터가 양수일 경우 MinMaxScaler 와 같습니다.
# 이상치에 매우 민감합니다.


from sklearn.preprocessing import MaxAbsScaler

from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
iris_dataset = load_iris()
X_train, X_test, y_train, y_test = train_test_split(
iris_dataset['data'], iris_dataset['target'], random_state=0)

# 변형 객체 생성
maxabs_scaler = MaxAbsScaler()

# 훈련데이터의 모수 분포 저장
maxabs_scaler.fit(X_train)

# 훈련 데이터 스케일링
X_train_scaled = maxabs_scaler.transform(X_train)

# 테스트 데이터의 스케일링
X_test_scaled = maxabs_scaler.transform(X_test)

# 스케일링 된 결과 값으로 본래 값을 구할 수도 있다.
# X_origin = maxabs_scaler.inverse_transform(X_train_scaled)

# [RobustScaler()]
# 평균과 분산 대신에 중간 값과 사분위 값을 사용합니다.
# 중간 값은 정렬시 중간에 있는 값을 의미하고
# 사분위값은 1/4, 3/4에 위치한 값을 의미합니다.
# 이상치 영향을 최소화할 수 있습니다.

from sklearn.preprocessing import RobustScaler

from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
iris_dataset = load_iris()
X_train, X_test, y_train, y_test = train_test_split(
iris_dataset['data'], iris_dataset['target'], random_state=0)

# 변형 객체 생성
robust_scaler = RobustScaler()

# 훈련데이터의 모수 분포 저장
robust_scaler.fit(X_train)

# 훈련 데이터 스케일링
X_train_scaled = robust_scaler.transform(X_train)

# 테스트 데이터의 스케일링
X_test_scaled = robust_scaler.transform(X_test)

# 스케일링 된 결과 값으로 본래 값을 구할 수도 있다.
# X_origin = robust_scaler.inverse_transform(X_train_scaled)


# Normalizer()
# 앞의 4가지 스케일러는 각 특성(열)의 통계치를 이용하여 진행됩니다.
# 그러나 Normalizer 의 경우 각 샘플(행)마다 적용되는 방식입니다.
# 이는 한 행의 모든 특성들 사이의 유클리드 거리(L2 norm)가 1이 되도록 스케일링합니다.
# 일반적인 데이터 전처리의 상황에서 사용되는 것이 아니라
# 모델(특히나 딥러닝) 내 학습 벡터에 적용하며,
# 특히나 피쳐들이 다른 단위(키, 나이, 소득 등)라면 더더욱 사용하지 않습니다

from sklearn.preprocessing import Normalizer

from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
iris_dataset = load_iris()
X_train, X_test, y_train, y_test = train_test_split(
iris_dataset['data'], iris_dataset['target'], random_state=0)

# 변형 객체 생성
normal_scaler = Normalizer()

# 훈련데이터의 모수 분포 저장
normal_scaler.fit(X_train)

# 훈련 데이터 스케일링
X_train_scaled = normal_scaler.transform(X_train)

# 테스트 데이터의 스케일링
X_test_scaled = normal_scaler.transform(X_test)

# 스케일링 된 결과 값으로 본래 값을 구할 수도 있다.
# X_origin = normal_scaler.inverse_transform(X_train_scaled)