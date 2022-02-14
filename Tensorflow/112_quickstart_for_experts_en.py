
# Set up TensorFlow
import tensorflow as tf
print("TensorFlow version:", tf.__version__)
# TensorFlow version: 2.3.0

# Load a dataset
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

# Build a machine learning model
model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(28, 28)),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(10)
])

# For each example, the model returns a vector of logits or log-odds scores,
#  one for each class.
predictions = model(x_train[:1]).numpy()
predictions
# array([[-0.61781377, -0.14540836,  0.08683631, -0.06089845,  0.24253306,
#          0.26507422, -0.15724811,  0.98673666, -0.46753916,  0.01557733]],
#       dtype=float32)

# The tf.nn.softmax function converts these logits to probabilities for each class:
tf.nn.softmax(predictions).numpy()
# array([[0.04816758, 0.07725345, 0.09744969, 0.08406594, 0.11386722,
#         0.11646306, 0.07634417, 0.23966366, 0.05597811, 0.09074716]],
#       dtype=float32)

# Define a loss function for training using losses.SparseCategoricalCrossentropy,
#  which takes a vector of logits and a True index and returns
#  a scalar loss for each example.
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

# This loss is equal to the negative log probability of the true class:
#  The loss is zero if the model is sure of the correct class.

# This untrained model gives probabilities close to random (1/10 for each class),
#  so the initial loss should be close to -tf.math.log(1/10) ~= 2.3.
loss_fn(y_train[:1], predictions).numpy()
2.1501813

# Before you start training, configure and compile the model using Keras Model.compile.
#  Set the optimizer class to adam, set the loss to the loss_fn function
#  you defined earlier, and specify a metric to be evaluated for the model by setting
#  the metrics parameter to accuracy.
model.compile(optimizer='adam',
              loss=loss_fn,
              metrics=['accuracy'])

# Train and evaluate your model
# Use the Model.fit method to adjust your model parameters and minimize the loss:
model.fit(x_train, y_train, epochs=5)
# Epoch 1/5
# 1875/1875 [==============================] - 1s 600us/step - loss: 0.2927 - accuracy: 0.9138
# Epoch 2/5
# 1875/1875 [==============================] - 1s 578us/step - loss: 0.1422 - accuracy: 0.9573
# Epoch 3/5
# 1875/1875 [==============================] - 1s 587us/step - loss: 0.1083 - accuracy: 0.9673
# Epoch 4/5
# 1875/1875 [==============================] - 1s 570us/step - loss: 0.0886 - accuracy: 0.9730
# Epoch 5/5
# 1875/1875 [==============================] - 1s 581us/step - loss: 0.0750 - accuracy: 0.9764
# <tensorflow.python.keras.callbacks.History at 0x1d2174f7a00>

# The Model.evaluate method checks the models performance,
#  usually on a "Validation-set" or "Test-set"
model.evaluate(x_test,  y_test, verbose=2)
# 313/313 - 0s - loss: 0.0779 - accuracy: 0.9759
# [0.07789715379476547, 0.9758999943733215]

# The image classifier is now trained to ~98% accuracy on this dataset.
#  To learn more, read the TensorFlow tutorials.

# If you want your model to return a probability, you can wrap the trained model,
#  and attach the softmax to it:
probability_model = tf.keras.Sequential([
  model,
  tf.keras.layers.Softmax()
])
probability_model(x_test[:5])
# <tf.Tensor: shape=(5, 10), dtype=float32, numpy=
# array([[3.8356855e-09, 9.7555613e-08, 2.5710856e-06, 3.4268736e-04,
#         9.3261170e-11, 1.9065893e-06, 2.8466685e-13, 9.9964726e-01,
#         1.8326635e-07, 5.2792038e-06],
#        [1.5640953e-07, 3.1444666e-05, 9.9996734e-01, 1.0262378e-06,
#         4.1090919e-15, 3.8121069e-08, 2.6874778e-08, 5.2912019e-12,
#         1.5161170e-08, 1.0097237e-13],
#        [2.3818046e-08, 9.9977857e-01, 3.7382226e-05, 3.4054570e-05,
#         1.1594939e-05, 1.0221799e-05, 5.3980798e-06, 6.5008506e-05,
#         5.6046236e-05, 1.8123063e-06],
#        [9.9969471e-01, 2.0735476e-08, 1.8911854e-04, 9.3461318e-07,
#         5.0068860e-07, 1.0011729e-05, 4.6994184e-05, 3.3133588e-06,
#         4.6017355e-08, 5.4479024e-05],
#        [5.9078019e-07, 8.6163601e-09, 3.6383672e-06, 1.2085228e-07,
#         9.9583900e-01, 1.1492036e-06, 3.0145438e-06, 6.7703090e-06,
#         2.8089044e-06, 4.1428283e-03]], dtype=float32)>

# Conclusion
# Congratulations! You have trained a machine learning model using
#  a prebuilt dataset using the Keras API.

# For more examples of using Keras, check out the tutorials.
#  To learn more about building models with Keras, read the guides.
#  If you want learn more about loading and preparing data,
#  see the tutorials on image data loading or CSV data loading.