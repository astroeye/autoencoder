# 텐서플로 2.0 시작하기: 초보자용

# 이 문서는 구글 코랩(Colaboratory) 노트북 파일입니다. 파이썬 프로그램을 브라우저에서 직접 실행할 수 있기 때문에 텐서플로를 배우고 사용하기 좋은 도구입니다:

# 파이썬 런타임(runtime)에 연결하세요: 메뉴 막대의 오른쪽 상단에서 CONNECT를 선택하세요.
# 노트북의 모든 코드 셀(cell)을 실행하세요: Runtime > Run all을 선택하세요.
# 더 많은 예제와 자세한 안내는 텐서플로 튜토리얼을 참고하세요.

# 먼저 프로그램에 텐서플로 라이브러리를 임포트합니다:

# !pip install -q tensorflow-gpu==2.0.0-rc1
import tensorflow as tf

# MNIST 데이터셋을 로드하여 준비합니다. 샘플 값을 정수에서 부동소수로 변환합니다:
mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

# 층을 차례대로 쌓아 tf.keras.Sequential 모델을 만듭니다. 훈련에 사용할 옵티마이저(optimizer)와 손실 함수를 선택합니다:
model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(28, 28)),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# 모델을 훈련하고 평가합니다:
model.fit(x_train, y_train, epochs=5)
model.evaluate(x_test,  y_test, verbose=2)

# Epoch 1/5
# 1875/1875 [==============================] - 1s 626us/step - loss: 0.2985 - accuracy: 0.9144
# Epoch 2/5
# 1875/1875 [==============================] - 1s 587us/step - loss: 0.1433 - accuracy: 0.9572
# Epoch 3/5
# 1875/1875 [==============================] - 1s 630us/step - loss: 0.1074 - accuracy: 0.9673
# Epoch 4/5
# 1875/1875 [==============================] - 1s 638us/step - loss: 0.0862 - accuracy: 0.9730
# Epoch 5/5
# 1875/1875 [==============================] - 1s 634us/step - loss: 0.0735 - accuracy: 0.9773
# 313/313 - 0s - loss: 0.0703 - accuracy: 0.9789
# [0.07028106600046158, 0.9789000153541565]

