import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Sequential 모델을 사용하는 경우
#     Sequential 모델은 각 레이어에 정확히 하나의 입력 텐서와 하나의 출력 텐서가 있는 일반 레이어 스택에 적합합니다.

#     개략적으로 다음과 같은 Sequential 모델은

# Define Sequential model with 3 layers
model = keras.Sequential(
    [
        layers.Dense(2, activation="relu", name="layer1"),
        layers.Dense(3, activation="relu", name="layer2"),
        layers.Dense(4, name="layer3"),
    ]
)
# Call model on a test input
x = tf.ones((3, 3))
y = model(x).

#     다음 함수와 동일합니다.

# Create 3 layers
layer1 = layers.Dense(2, activation="relu", name="layer1")
layer2 = layers.Dense(3, activation="relu", name="layer2")
layer3 = layers.Dense(4, name="layer3")

# Call layers on a test input
x = tf.ones((3, 3))
y = layer3(layer2(layer1(x)))

# Sequential 모델은 다음의 경우에 적합하지 않습니다.

#     모델에 다중 입력 또는 다중 출력이 있습니다
#     레이어에 다중 입력 또는 다중 출력이 있습니다
#     레이어 공유를 해야 합니다
#     비선형 토폴로지를 원합니다(예: 잔류 연결, 다중 분기 모델)

