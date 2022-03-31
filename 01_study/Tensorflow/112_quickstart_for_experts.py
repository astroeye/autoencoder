# 전문가를 위한 TensorFlow 2 빠른 시작

# 이것은 Google Colaboratory 노트북 파일입니다. Python 프로그램은
#  브라우저에서 직접 실행되므로 TensorFlow를 배우고 사용하기에 좋습니다.
#  이 튜토리얼을 따르려면 이 페이지 상단에 있는 버튼을 클릭하여
#  Google Colab에서 노트북을 실행하세요.

# 파이썬 런타임(runtime)에 연결하세요: 메뉴 막대의 오른쪽 상단에서
#  CONNECT를 선택하세요.
# 노트북의 모든 코드 셀(cell)을 실행하세요: Runtime > Run all을 선택하세요.
# TensorFlow 2를 다운로드하여 설치합니다. TensorFlow를 프로그램으로 가져옵니다.

# 참고: TensorFlow 2 패키지를 설치하려면 pip를 업그레이드하세요.
#  자세한 내용은 설치 가이드를 참조하세요.

# TensorFlow를 프로그램으로 가져옵니다.
import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten, Conv2D
from tensorflow.keras import Model

# MNIST 데이터셋을 로드하여 준비합니다.
mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

# Add a channels dimension
x_train = x_train[..., tf.newaxis].astype("float32")
x_test = x_test[..., tf.newaxis].astype("float32")

# tf.data를 사용하여 데이터셋을 섞고 배치를 만듭니다:
train_ds = tf.data.Dataset.from_tensor_slices(
    (x_train, y_train)).shuffle(10000).batch(32)

test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(32)

# 케라스(Keras)의 모델 서브클래싱(subclassing) API를 사용하여
#  tf.keras 모델을 만듭니다:
class MyModel(Model):
  def __init__(self):
    super(MyModel, self).__init__()
    self.conv1 = Conv2D(32, 3, activation='relu')
    self.flatten = Flatten()
    self.d1 = Dense(128, activation='relu')
    self.d2 = Dense(10)

  def call(self, x):
    x = self.conv1(x)
    x = self.flatten(x)
    x = self.d1(x)
    return self.d2(x)

# Create an instance of the model
model = MyModel()

# 훈련에 필요한 옵티마이저(optimizer)와 손실 함수를 선택합니다:
loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
optimizer = tf.keras.optimizers.Adam()

# 모델의 손실과 성능을 측정할 지표를 선택합니다.
#  에포크가 진행되는 동안 수집된 측정 지표를 바탕으로 최종 결과를 출력합니다.
train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')

test_loss = tf.keras.metrics.Mean(name='test_loss')
test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')

# tf.GradientTape를 사용하여 모델을 훈련합니다:
@tf.function
def train_step(images, labels):
  with tf.GradientTape() as tape:
    # training=True is only needed if there are layers with different
    # behavior during training versus inference (e.g. Dropout).
    predictions = model(images, training=True)
    loss = loss_object(labels, predictions)
  gradients = tape.gradient(loss, model.trainable_variables)
  optimizer.apply_gradients(zip(gradients, model.trainable_variables))

  train_loss(loss)
  train_accuracy(labels, predictions)

# 이제 모델을 테스트합니다:
@tf.function
def test_step(images, labels):
  # training=False is only needed if there are layers with different
  # behavior during training versus inference (e.g. Dropout).
  predictions = model(images, training=False)
  t_loss = loss_object(labels, predictions)

  test_loss(t_loss)
  test_accuracy(labels, predictions)

EPOCHS = 5

for epoch in range(EPOCHS):
  # Reset the metrics at the start of the next epoch
  train_loss.reset_states()
  train_accuracy.reset_states()
  test_loss.reset_states()
  test_accuracy.reset_states()

  for images, labels in train_ds:
    train_step(images, labels)

  for test_images, test_labels in test_ds:
    test_step(test_images, test_labels)

  print(
    f'Epoch {epoch + 1}, '
    f'Loss: {train_loss.result()}, '
    f'Accuracy: {train_accuracy.result() * 100}, '
    f'Test Loss: {test_loss.result()}, '
    f'Test Accuracy: {test_accuracy.result() * 100}'
  )

# Epoch 1, Loss: 0.13157518208026886, Accuracy: 95.97833251953125, Test Loss: 0.07298742979764938, Test Accuracy: 97.52999877929688
# Epoch 2, Loss: 0.04080909118056297, Accuracy: 98.6883316040039, Test Loss: 0.04941999539732933, Test Accuracy: 98.36000061035156
# Epoch 3, Loss: 0.019416792318224907, Accuracy: 99.36000061035156, Test Loss: 0.05908839777112007, Test Accuracy: 98.19999694824219
# Epoch 4, Loss: 0.012170523405075073, Accuracy: 99.58333587646484, Test Loss: 0.061041779816150665, Test Accuracy: 98.3699951171875
# Epoch 5, Loss: 0.009129365906119347, Accuracy: 99.69000244140625, Test Loss: 0.05872734263539314, Test Accuracy: 98.5199966430664

# 훈련된 이미지 분류기는 이 데이터셋에서 약 98%의 정확도를 달성합니다.
#  더 자세한 내용은 TensorFlow 튜토리얼을 참고하세요.
