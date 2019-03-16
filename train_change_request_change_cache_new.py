#!/usr/bin/env python
from __future__ import print_function
from scipy.special import comb

import tensorflow as tf
import random
import numpy as np
#import matplotlib.pyplot as plt
import argparse
from collections import deque
from itertools import combinations
from tensorflow.python.framework import ops
ops.reset_default_graph()

GAMMA = 1 # decay rate of past observations
OBSERVE = 20000. # timesteps to observe before training
EXPLORE = 3000000. # frames over which to anneal epsilon
FINAL_EPSILON = 0.0001 # final value of epsilon
INITIAL_EPSILON = 0.1 # starting value of epsilon
REPLAY_MEMORY = 50000 # number of previous transitions to remember
BATCH = 32 # size of minibatch
STEP_PER_ACTION = 1
LEARNING_RATE = 1e-4
TARGET_FUNCTION = 4
STABLE_CACHE = 0
MAX_ITERATION = 10000000
NODE_NUM = 64
QUICK = 100
OBSERVE = OBSERVE / QUICK
EXPLORE = EXPLORE / QUICK


# Network parameters
parser = argparse.ArgumentParser(description='manual to this script')
parser.add_argument('--K', type=int, default = 10)
parser.add_argument('--N', type=int, default = 20)
parser.add_argument('--M', type=int, default = 3)
parser.add_argument('--L', type=int, default = 2)
args = parser.parse_args()
print(args.K)
print(args.N)
print(args.M)
print(args.L)
K = args.K
N = args.N
M = args.M
L = args.L
#K = 5  # user number
#N = 20  # file number
#M = 3  # cache size of edge server
#L = 2  # cache size of user
ACTIONS = int(comb(N, M)) # number of valid actions
#ACTIONS = N
files = [n for n in range(1, N + 1)]
cob = list(combinations(files, M))  # cache state index starts from 0

# generate the transition probability
ga = 0.5;
Q0 = 0.2;
nn = 1;
f =np.array([[0 for col in range(N + 1)] for row in range(N + 1)], dtype = float)
f_sum = np.array([[0 for col in range(N + 1)] for row in range(N + 1)], dtype = float)
sum_pro = 0;
for i in range(1, N + 1):
    sum_pro = sum_pro + 1 / i ** ga
for i in range(0, N + 1):
    for j in range(0, N + 1): 
        if j == 0:
            f[:, j] = Q0 * np.ones(N + 1)
            continue
        if i == 0:
            f[i, j] = (1 - Q0) * (1 / j ** ga / sum_pro)
        #elif j == i % N + 1 or j == (i + 1) % N + 1:
        elif j == i % N + 1:
            f[i, j] = (1 - Q0) / nn;

for i in range(0, N + 1):
    for j in range(0, N + 1):
        f_sum[i, j] = f[i, 0 : j + 1].sum()      

def cache_diff(old_cache_state, new_cache_state):
    add = list(set(new_cache_state).difference(set(old_cache_state)))
    delete = list(set(old_cache_state).difference(set(new_cache_state)))
    maintain = list(set(old_cache_state).intersection(set(new_cache_state)))
    return add, delete, maintain

class Environment(object):
    def __init__(self, request, cache):
        self.request_state = request
        self.cache_state = cache
        self.old_cache_state = cache
        self.push = 0
        self.reward = 0
        
    
    def _step(self, cache_action):
        # compute the reactive transmission, push and cost
        R = list(set(self.request_state).difference(set(self.cache_state)))
        if len(R):
            if R[0] == 0:
                del(R[0])
        
        self.old_cache_state = self.cache_state
        self.cache_state = list(cob[cache_action])
        add = list(set(self.cache_state).difference(set(self.old_cache_state)))
        P = list(set(add).difference(set(R)))
        # cost_0 = ((len(R) + len(P)) / min(N, M + K)) ** TARGET_FUNCTION
        cost_0 = (len(R) + len(P)) ** TARGET_FUNCTION
        
        # changes of user requests
        for i in range(0, K):
            tran_p = random.random()
            j = 0;
            while tran_p > f_sum[self.request_state[i], j]:
                j += 1
            self.request_state[i] = j
        
        return self.request_state, cost_0, R, P, self.old_cache_state
        
    def _random_step(self):
        # compute the reactive transmission, push and cost
        R = list(set(self.request_state).difference(set(self.cache_state)))
        if len(R):
            if R[0] == 0:
                del(R[0])
        self.old_cache_state = self.cache_state
        cache_action = random.randint(0, ACTIONS - 1)
        self.cache_state = list(cob[cache_action])
        add = list(set(self.cache_state).difference(set(self.old_cache_state)))
        P = list(set(add).difference(set(R)))
        #cost_0 = ((len(R) + len(P)) / min(N, M + K)) ** TARGET_FUNCTION
        cost_0 = (len(R) + len(P)) ** TARGET_FUNCTION - 0 ** TARGET_FUNCTION
        
        # changes of user requests
        for i in range(0, K):
            tran_p = random.random()
            j = 0;
            while tran_p > f_sum[self.request_state[i], j]:
                j += 1
            self.request_state[i] = j
            
        return self.request_state, cost_0, R, P
        

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev = 0.01)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.01, shape = shape)
    return tf.Variable(initial)

#def max_pool_2x2(x):
#    return tf.nn.max_pool(x, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = "SAME")

def createNetwork():
    # network weights
    W_conv1 = weight_variable([N + N + 2, NODE_NUM])
    b_conv1 = bias_variable([NODE_NUM])

    W_conv2 = weight_variable([NODE_NUM, NODE_NUM])
    b_conv2 = bias_variable([NODE_NUM])

#    W_conv3 = weight_variable([3, 3, 64, 64])
#    b_conv3 = bias_variable([64])

    W_fc1 = weight_variable([NODE_NUM, NODE_NUM])
    b_fc1 = bias_variable([NODE_NUM])

    W_fc2 = weight_variable([NODE_NUM, ACTIONS])
    b_fc2 = bias_variable([ACTIONS])

    # input layer
    s = tf.placeholder("float", [None, N + N + 2])

    # hidden layers
    h_conv1 = tf.nn.relu(tf.add(tf.matmul(s, W_conv1), b_conv1))

#    h_conv2 = tf.nn.relu(tf.add(tf.matmul(h_conv1, W_conv2), b_conv2))

#    h_conv3 = tf.add(tf.matmul(h_conv2, W_conv3), b_conv3)

#    h_conv4 = tf.reshape(h_conv3, [-1, 1600])

    #h_fc1 = tf.nn.relu(tf.add(tf.matmul(h_conv2, W_fc1), b_fc1))
    h_fc1 = tf.nn.relu(tf.add(tf.matmul(h_conv1, W_fc1), b_fc1))

    # readout layer
    readout = tf.nn.relu(tf.add(tf.matmul(h_fc1, W_fc2), b_fc2))

    return s, readout

def trainNetwork(s, readout, sess, P_u, R_u):
    # define the cost function
    a = tf.placeholder("float", [None, ACTIONS])
    y = tf.placeholder("float", [None])
    readout_action = tf.reduce_sum(tf.multiply(readout, a), reduction_indices=1)
    #readout_action = tf.multiply(readout, a)
    cost = tf.reduce_mean(tf.square(y - readout_action))
    train_step = tf.train.AdamOptimizer(LEARNING_RATE).minimize(cost)

    # open up a system environment
    request = np.zeros(K)
    for i in range(0, K):
        request[i] = random.randint(0, N)
    request = request.astype(int)
    old_request_state = request.copy()
    cache = files
    random.shuffle(cache)
    cache = cache[0 : M]
    env = Environment(request, cache)
    R = list(set(request).difference(set(cache)))
    if len(R):
        if R[0] == 0:
            del(R[0])
    RL = [len(R)]
    request_num = np.zeros(N + 1)
    for i in range(0, K):
        request_num[request[i]] += 1
    cache.sort()
    cache_index = np.zeros(N)
    for i in cache:
        cache_index[i - 1] += 1
    s_t = np.array(list(request_num) + list(cache_index) + RL)
    s_0 = s_t

    # store the previous observations in replay memory
    D = deque()

    # get the first state by doing nothing and preprocess the image to 80x80x4
#    s_t = np.array(request + cache)
#    rand_action = random.randint(0, ACTIONS - 1)
#    rand_cache = np.zeros(ACTIONS)
#    rand_cache[rand_action] = 1
#    old_request = request
#    request, cost_0 = env._step(rand_cache)
#    s_t1 = np.array(request + cache)

    # saving and loading networks
    saver = tf.train.Saver()
    sess.run(tf.initialize_all_variables())
#    checkpoint = tf.train.get_checkpoint_state("saved_networks")
    """
    if checkpoint and checkpoint.model_checkpoint_path:
        saver.restore(sess, checkpoint.model_checkpoint_path)
        print("Successfully loaded:", checkpoint.model_checkpoint_path)
    else:
        print("Could not find old network weights")
    """
    # start training
    epsilon = INITIAL_EPSILON
    t = 0
    total_cost = 0
    LOSS_FUN = []
    
    while t <= MAX_ITERATION:
        if t >= 1:
            old_request_state = new_request_state.copy()
        # choose an action epsilon greedily
        readout_t = readout.eval(feed_dict={s : [s_t]})[0]
        a_t = np.zeros([ACTIONS])
        action_index = 0
        #if t % STEP_PER_ACTION == 0:
        if random.random() <= epsilon:
#            print("----------Random Action----------")
            if STABLE_CACHE == 0:
                action_index = random.randrange(ACTIONS)
            a_t[action_index] = 1
        else:
            if STABLE_CACHE == 0:
                #action_index = np.argmin(readout_t)
                min_index = np.where(readout_t == np.min(readout_t))
                min_index = min_index[0]
                action_index = min_index[random.randint(0, len(min_index)-1)]
            a_t[action_index] = 1
        #else:
        #    a_t[0] = 1 # do nothing

        # scale down epsilon
        if epsilon > FINAL_EPSILON and t > OBSERVE:
            epsilon -= (INITIAL_EPSILON - FINAL_EPSILON) / EXPLORE

        # run the selected action and observe next state and reward
        new_request_state, r_t, R, P, old_cache = env._step(action_index)
        P_uo = old_request_state.copy() % 10 + 1
        P_u = []
        for i in range(0, K):
            if P_uo[i] not in old_cache:
                P_u.append(P_uo[i])
        R_u = old_request_state.copy()
        r_t += (len(P_u) + len(R_u)) ** TARGET_FUNCTION
        RL = [len(R)]
        new_request_num = np.zeros(N + 1)
        for i in range(0, K):
            new_request_num[request[i]] += 1
        old_cache = cache.copy()
        cache = list(cob[action_index])
        # count the different cache nums
        #cache_diff = len(set(cache).difference(set(old_cache)))
        
        cache_index = np.zeros(N)
        for i in cache:
            cache_index[i - 1] += 1
        s_t1 = np.array(list(new_request_num) + list(cache_index) + RL)
        #total_cost += r_t * (min(N, M + K) ** TARGET_FUNCTION)
        total_cost += r_t
        avarage_cost = total_cost / (t + 1)

        # store the transition in D
        D.append((s_t, a_t, r_t, s_t1))
        if len(D) > REPLAY_MEMORY:
            D.popleft()

        # only train if done observing
        if t > OBSERVE:
            # sample a minibatch to train on
            minibatch = random.sample(D, BATCH)
            #print (minibatch.shape)
            # get the batch variables
            s_j_batch = [d[0] for d in minibatch]
            #print (s_j_batch.shape)
            a_batch = [d[1] for d in minibatch]
            #print (a_batch.shape)
            r_batch = [d[2] for d in minibatch]
            #print (r_batch.shape)
            s_j1_batch = [d[3] for d in minibatch]
            #print (s_j1_batch.shape)

            y_batch = []
            readout_j1_batch = readout.eval(feed_dict = {s : s_j1_batch})
            readout_j0 = readout.eval(feed_dict = {s : [s_0]})[0]
            for i in range(0, len(minibatch)):
                y_batch.append(r_batch[i] + GAMMA * np.min(readout_j1_batch[i]) - np.min(readout_j0))

            # perform gradient step
            sess.run(train_step, feed_dict = {y : y_batch, a : a_batch, s : s_j_batch})
#            train_step.run(feed_dict = {y : y_batch, a : a_batch, s : s_j_batch})
            LOSS = sess.run(cost, feed_dict = {y : y_batch, a : a_batch, s : s_j_batch})
            if t % 1000 == 0:
                print("/ LOSS", LOSS)
                print("P", P, "R", R, "SYSTEM_STATE", s_t, "NEXT_STATE", s_t1)
            
            LOSS_FUN.append(LOSS)

        # save progress every 10000 iterations
        if t % 10000 == 0:
            saver.save(sess, 'saved_networks/' + 'cache' + '-dqn', global_step = t)

        if t % (MAX_ITERATION / 10) == 0:
            str1 = "saved_networks_{KK}_{NN}_{MM}_{LL}_edge_server/cached-dqn-".format(KK=K, NN=N, MM=M, LL=L)
            saver.save(sess, str1 , global_step = t)
            
        # print info
#        state = ""
#        if t <= OBSERVE:
#            state = "observe"
#        elif t > OBSERVE and t <= OBSERVE + EXPLORE:
#            state = "explore"
#        else:
#            state = "train"
            
        if t % 1000 == 0:
            print("TIMESTEP", t, "/ AVARAGE_COST", avarage_cost, \
                  "/ ACTION", action_index, "/ REWARD", r_t)
        
        # update the old values
        s_t = s_t1
        t += 1
        # write info to files
        '''
        if t % 10000 <= 100:
            a_file.write(",".join([str(x) for x in readout_t]) + '\n')
            h_file.write(",".join([str(x) for x in h_fc1.eval(feed_dict={s:[s_t]})[0]]) + '\n')
            cv2.imwrite("logs_tetris/frame" + str(t) + ".png", x_t1)
        '''        
    return LOSS_FUN

def playGame(P_u, R_u):
    sess = tf.InteractiveSession()
    s, readout = createNetwork()
    loss = trainNetwork(s, readout, sess, P_u, R_u)
    return loss

def main():
    P_u = 1
    R_u = 1
    loss = playGame(P_u, R_u)
#    x = range(0, len(loss))
#    plt.plot(x, loss)

if __name__ == "__main__":
    main()
