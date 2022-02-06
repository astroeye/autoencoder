# https://wooono.tistory.com/94?category=918936
# [ML] Epoch, Batch size, Iterations 용어 정리

# [Epoch]
# 모든 훈련 데이터셋을 학습하는 횟수입니다.
# 주의 사항
#  너무 많은 Epoch는 overfitting의 위험이 있습니다.

# [Batch Size]
# 훈련 데이터셋 중 몇 개의 데이터를 묶어서 가중치 값을 갱신할 것인지에 대한 정보입니다.
# 위 그림처럼, 훈련 데이터셋의 개수는 1000개이며, 1 Epoch를 진행한다고 했을 때
# Batch Size를 100으로 설정하면 총 10번의 갱신 과정을 거칩니다.
# 주의 사항
#  Batch Size는 메모리에 적재시킬 수 있을 만큼의 Size로 정하는게 좋습니다.
#  Batch Size를 너무 작게하면 iteration이 증가하여 학습시간
#  (forward + backward propagation)이 오래 걸릴 수 있습니다.

# [Iterations]
# 한 Epoch를 진행하기 위해, 몇 번의 가중치 갱신이 이루어지는지에 대한 정보입니다.
# 위의 예시와 동일하게 훈련 데이터셋의 개수는 1000개이며, 1 Epoch를 진행한다고 했을 때
# Batch Size를 100으로 설정하면 총 10번의 갱신 과정을 거칩니다.
# 즉, iterations은 10번 입니다.