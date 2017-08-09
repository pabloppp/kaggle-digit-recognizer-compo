import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf
import sys

def normalize(v):
    return v / 255

data_x = list(map(normalize, map(int, sys.argv[1].split(','))))

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

saver = tf.train.import_meta_graph('model/trained_nn.ckpt.meta')
saver.restore(sess, 'model/trained_nn.ckpt')

perceptron = tf.get_collection('perceptron')[0]

inferences = sess.run(perceptron, {x: data_x}).tolist()
inferences = list(map(unitArrayToNum, inferences))

print(inferences[0])
