import tensorflow as tf

hello_data = 'Hello '
world_data = 'World!'

hello = tf.constant(hello_data)
world = tf.constant(world_data)
concat = hello + world
print(concat)

session = tf.Session()
print(session.run(concat))