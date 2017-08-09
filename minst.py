import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import csv
import tensorflow as tf
import numpy as np
import time
import random

# --- GROUND DATA ADQUISITION

x_training = []
y_training = []

def numToUnitArray(num):
    arr = [0] * 10
    arr[num] = 1
    return arr

def unitArrayToNum(arr):
    return arr.index(max(arr))

def normalize(num):
    return num  / 255.

def accuracy(arr1, arr2):
    return np.mean( np.array(arr1) == np.array(arr2) )

gStartTime = time.time()
print("(^__^)/ -- Reading ground truth data in memory...")
with open('train.csv', 'r') as f:
    reader = csv.reader(f)
    next(reader, None)
    for row in reader:
        y_training.append(numToUnitArray(int(row[0])))
        x_training.append(list(map(normalize, map(int, row[1:785]))))

# print("(O__o) -- Reading expanded ground truth data in memory...")
# with open('reshaped.csv', 'r') as f:
#     reader = csv.reader(f)
#     for row in reader:
#         y_training.append(numToUnitArray(int(row[0])))
#         x_training.append(list(map(normalize, map(int, row[1:785]))))

print("Read in", time.time() - gStartTime, "sec")
print("Labels count:", len(y_training))
## print("Label example:", y_training[0])
print("Imgs count:", len(x_training))
print("Img pixel count:", len(x_training[0]))
#print("Img example:", x_training[0][:300])

# --- FCNN

N_INPUT_NODES = 784
N_HIDDEN_NODES = 784 # 200
N_OUTPUT_NODES = 10

N_EPOCH = 20
ITERATIONS_COUNT = 800 # 800

rate = 0.001

x = tf.placeholder(tf.float32, [None, N_INPUT_NODES])
y = tf.placeholder(tf.float32, [None, N_OUTPUT_NODES])

b1 = tf.Variable(tf.zeros([N_HIDDEN_NODES]))
b2 = tf.Variable(tf.zeros([N_OUTPUT_NODES]))

w1 = tf.Variable(tf.random_uniform([N_INPUT_NODES, N_HIDDEN_NODES], -1, 1))
w2 = tf.Variable(tf.random_uniform([N_HIDDEN_NODES, N_OUTPUT_NODES], -1, 1))

l1 = tf.nn.relu(tf.matmul(x, w1) + b1) # tf.sigmoid(tf.matmul(x, w1) + b1)
perceptron = tf.nn.softmax(tf.matmul(l1, w2) + b2)

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=perceptron, labels=y)) #- tf.reduce_mean( (y * tf.log(perceptron)) + (1 - y) * tf.log(1.0001 - perceptron)  )
optimizer = tf.train.AdamOptimizer(rate)
train = optimizer.minimize(cost)

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

saver = tf.train.Saver()

# saver = tf.train.import_meta_graph('model/trained_nn.ckpt.meta')
# saver.restore(sess, 'model/trained_nn.ckpt')

print("(ò__-)- -- Ready to train!!...")
gStartTime = time.time()
iStartTime = time.time()
for i in range(ITERATIONS_COUNT):
    sess.run(train, {x: x_training[12600:], y: y_training[12600:]})
    if i % N_EPOCH == 0:
        print('Batch ', i, '/', ITERATIONS_COUNT)
        inferences = sess.run(perceptron, {x: x_training[:12600], y: y_training[:12600]}).tolist()
        inferences = list(map(unitArrayToNum, inferences))
        expected = list(map(unitArrayToNum, y_training))
        rand = random.randint(0, 5000)
        print(' expected:   ', expected[rand:rand+15])
        print(' prediction: ', inferences[rand:rand+15])
        print(' accuracy:   ', accuracy(inferences[:5000], expected[:5000]))
        print(' Cost:    ', sess.run(cost, {x: x_training[:12600], y: y_training[:12600]}))
        print(' Elapsed time (since last batch):', time.time() - gStartTime, 's')
        gStartTime = time.time()

tf.add_to_collection('perceptron', perceptron)
save_path = saver.save(sess, 'model/trained_nn.ckpt')

print('(^ 3 ^)´ -- DONE in ', time.time() - iStartTime ,'s!!')

inferences = sess.run(perceptron, {x: x_training[:12600], y: y_training[:12600]}).tolist()
inferences = list(map(unitArrayToNum, inferences))
expected = list(map(unitArrayToNum, y_training))
rand = random.randint(0, 5000)
print(' expected:   ', expected[rand:rand+30])
print(' prediction: ', inferences[rand:rand+30])
print(' accuracy:   ', accuracy(inferences[:12000], expected[:12000]))

# print("(^__^)/ -- Saving weights...")
# np.savetxt('b1.txt', b1)
# np.savetxt('b2.txt', b2)
# np.savetxt('w1.txt', w1)
# np.savetxt('w2.txt', w2)

x_test = []

print("(^__^)/ -- Reading test data in memory...")
with open('test.csv', 'r') as f:
    reader = csv.reader(f)
    next(reader, None)
    for row in reader:
        x_test.append(list(map(normalize, map(int, row[0:784]))))

print("(o__O)~ -- Calculating submission...")
submissions = sess.run(perceptron, {x: x_test }).tolist()
submissions = list(map(unitArrayToNum, submissions))
print("submission count:", len(submissions))

print("(- 3 O), -- Writing submission!")
with open('submission.csv', 'w') as f:
    f.write("ImageId,Label\n")
    i = 1
    for s in submissions:
        f.write(str(i))
        f.write(",")
        f.write(str(s))
        f.write("\n")
        i += 1

print('(^ 3 ^)´ -- DONE AGAIN!')
