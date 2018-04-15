import tensorflow as tf
import numpy as np

x_data = np.linspace(0., 1., 6)
a_answer = 1.5
b_answer = .1
y_data = a_answer * x_data + b_answer

x = tf.placeholder(tf.float32)
y_answer = tf.placeholder(tf.float32)

a_model = tf.Variable(1.0)
b_model = tf.Variable(0.0)

y_model = a_model * x + b_model
loss = tf.sqrt(tf.reduce_mean((y_model - y_answer) ** 2))
train = tf.train.GradientDescentOptimizer(0.0001).minimize(loss)

init = tf.global_variables_initializer()


session = tf.Session()
session.run(init)

for i in range(20000):
    session.run(train, {x: x_data, y_answer: y_data})
    if i % 1000 == 0:
        current_loss, current_y_model = session.run(
            [loss, y_model], {x: x_data, y_answer: y_data}
        )
        print(f"Loss: {current_loss}")
        print(f"y_model: {current_y_model}, y_answer: {y_data}")
