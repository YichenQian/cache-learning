#!/usr/bin/env python
from __future__ import print_function
from scipy.special import comb
from collections import Counter

import math
import tensorflow as tf
import random
import numpy as np
#import matplotlib.pyplot as plt
import argparse
#from collections import deque
from itertools import combinations
from tensorflow.python.framework import ops
ops.reset_default_graph()

STEP_PER_ACTION = 1
TARGET_FUNCTION = 4
STABLE_CACHE = 0
MAX_ITERATION = 10000
NODE_NUM = 64

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
#N = 20 # file number
#M = 3  # cache size
#L = 2
ACTIONS = int(comb(N, M)) # number of valid actions
#ACTIONS = N
files = [n for n in range(1, N + 1)]
cob = list(combinations(files, M))  # cache state index starts from 0

# generate the transition probability
ga = 0.5
Q0 = 0.2
nn = 1
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
        
sigmap = np.sum(f[0, :] ** 2)

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
            R.sort()
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
        
        return self.request_state, cost_0, R, P
        
    def _random_step(self):
        # compute the reactive transmission, push and cost
        R = list(set(self.request_state).difference(set(self.cache_state)))
        if len(R):
            R.sort()
            if R[0] == 0:
                del(R[0])
        self.old_cache_state = self.cache_state
        cache_action = random.randint(0, ACTIONS - 1)
        self.cache_state = list(cob[cache_action])
        add = list(set(self.cache_state).difference(set(self.old_cache_state)))
        P = list(set(add).difference(set(R)))
        #cost_0 = ((len(R) + len(P)) / min(N, M + K)) ** TARGET_FUNCTION
        cost_0 = (len(R) + len(P)) ** TARGET_FUNCTION
        
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

def conv1d(x, W, stride):
    return tf.nn.conv2d(x, W, strides = [1, stride, stride, 1], padding = "SAME")

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

def trainNetwork(s, readout, sess):
    # define the cost function
    a = tf.placeholder("float", [None, ACTIONS])
    y = tf.placeholder("float", [None])

    # open up a system environment
    request = np.zeros(K)
    for i in range(0, K):
        request[i] = random.randint(0, N)
    request = request.astype(int)
    cache = files
    random.shuffle(cache)
    cache = cache[0 : M]
    cache.sort()
    env = Environment(request, cache)
    R = list(set(request).difference(set(cache)))
    if len(R):
        R.sort()
        if R[0] == 0:
            del(R[0])
    RL = [len(R)]
    request_num = np.zeros(N + 1)
    for i in range(0, K):
        request_num[request[i]] += 1
    cache_index = np.zeros(N)
    for i in cache:
        cache_index[i - 1] += 1
    s_t = np.array(list(request_num) + list(cache_index) + RL)
    

    # get the first state by doing nothing and preprocess the image to 80x80x4
#    s_t = np.array(request + cache)
#    rand_action = random.randint(0, ACTIONS - 1)
#    rand_cache = np.zeros(ACTIONS)
#    rand_cache[rand_action] = 1
#    old_request = request
#    request, cost_0 = env._step(rand_cache)
#    s_t1 = np.array(request + cache)

    # saving and loading networks
    str1 = "saved_networks_{KK}_{NN}_{MM}_{LL}_edge_server/cached-dqn-".format(KK=K, NN=N, MM=M, LL=L)
    saver = tf.train.Saver()
    sess.run(tf.initialize_all_variables())
    checkpoint = tf.train.get_checkpoint_state(str1)
    if checkpoint and checkpoint.model_checkpoint_path:
        saver.restore(sess, checkpoint.model_checkpoint_path)
        print("Successfully loaded:", checkpoint.model_checkpoint_path)
    else:
        print("Could not find old network weights")
#    checkpoint = tf.train.get_checkpoint_state("saved_networks")
    """
    if checkpoint and checkpoint.model_checkpoint_path:
        saver.restore(sess, checkpoint.model_checkpoint_path)
        print("Successfully loaded:", checkpoint.model_checkpoint_path)
    else:
        print("Could not find old network weights")
    """
    # start training
    t = 0
    total_cost = 0
    push_times = 0
    push_files = 0
    
    while t < MAX_ITERATION:
        # choose an action epsilon greedily
        readout_t = readout.eval(feed_dict={s : [s_t]})[0]
        a_t = np.zeros([ACTIONS])
        action_index = 0
        #if t % STEP_PER_ACTION == 0:
        if STABLE_CACHE == 0:
            action_index = random.randrange(ACTIONS)
        a_t[action_index] = 1
        if STABLE_CACHE == 0:
            #action_index = np.argmin(readout_t)
            min_index = np.where(readout_t == np.min(readout_t))
            min_index = min_index[0]
            action_index = min_index[random.randint(0, len(min_index)-1)]
        a_t[action_index] = 1

        # run the selected action and observe next state and reward
        new_request_state, r_t, R, P = env._step(action_index)
        if P:
            push_times += 1
            push_files += len(P)
        RL = [len(R)]
        new_request_num = np.zeros(N + 1)
        for i in range(0, K):
            new_request_num[request[i]] += 1
        cache = cob[action_index]
        cache_index = np.zeros(N)
        for i in cache:
            cache_index[i - 1] += 1
        s_t1 = np.array(list(new_request_num) + list(cache_index) + RL)
        #total_cost += r_t * (min(N, M + K) ** TARGET_FUNCTION)
        total_cost += r_t
        avarage_cost = total_cost / (t + 1)

#        if t % 1000 == 0:
#            print("TIMESTEP", t, "/ AVARAGE_COST", avarage_cost, \
#                  "/ ACTION", action_index, "/ REWARD", r_t)
        
        # update the old values
        s_t = s_t1
        t += 1
        if t % 1000 == 0:
            print("RL_t:", t)
        # write info to files
        '''
        if t % 10000 <= 100:
            a_file.write(",".join([str(x) for x in readout_t]) + '\n')
            h_file.write(",".join([str(x) for x in h_fc1.eval(feed_dict={s:[s_t]})[0]]) + '\n')
            cv2.imwrite("logs_tetris/frame" + str(t) + ".png", x_t1)
        '''        
    return avarage_cost, push_times, push_files

def playGame():
    sess = tf.InteractiveSession()
    s, readout = createNetwork()
    avarage_cost, push_times, push_files = trainNetwork(s, readout, sess)
    return avarage_cost, push_times, push_files

def LRU():
    # open up a system environment
    request = np.zeros(K)
    for i in range(0, K):
        request[i] = random.randint(0, N)
    request = request.astype(int)
    cache = files
    random.shuffle(cache)
    cache = cache[0 : M]
    cache.sort()
    env = Environment(request, cache)
    new_request_state = request
    total_cost = 0
    t = 0
    while t < MAX_ITERATION:
        R = list(set(new_request_state).difference(set(cache)))
        if len(R):
            R.sort()
            if R[0] == 0:
                del(R[0])
        # LRU update
        if len(R) >= M:
            random.shuffle(R)
            cache = R[0 : M]
        else:
            for i in range(K):
                if new_request_state[i] in cache:
                    ind = cache.index(new_request_state[i])
                    cache[1 : ind + 1] = cache[0 : ind]
                    cache[0] = new_request_state[i]
                elif new_request_state[i] != 0:
                    cache[1 :] = cache[0 : -1]
                    cache[0] = new_request_state[i]
        cache.sort()
        action_index = cob.index(tuple(cache))
        new_request_state, r_t, R, P = env._step(action_index)
        total_cost += len(R) ** TARGET_FUNCTION
        t += 1
        if t % 1000 == 0:
            print("LRU:", t)
    avarage_cost = total_cost / t
    return avarage_cost

def LFU():
    # open up a system environment
    request = np.zeros(K)
    for i in range(0, K):
        request[i] = random.randint(0, N)
    request = request.astype(int)
    cache = files
    random.shuffle(cache)
    cache = cache[0 : M]
    cache.sort()
    env = Environment(request, cache)
    total_cost = 0
    t = 0
    count_freq = np.zeros(M)
    new_request_state = request
    while t < MAX_ITERATION:
        R = list(set(new_request_state).difference(set(cache)))
        if len(R):
            R.sort()
            if R[0] == 0:
                del(R[0])
        # LFU update
        for i in range(K):
            if new_request_state[i] in cache:
                count_freq[cache.index(new_request_state[i])] += 1
        index = []
        if len(R) >= M:
            random.shuffle(R)
            cache = R[0 : M]
        else:
            for i in range(len(R)):
                ind = np.argmin(count_freq)
                index.append(ind)
                count_freq[ind] = 1000
                cache[ind] = R[i]
            for i in index:    
                count_freq[i] = 1
        cache.sort()
        action_index = cob.index(tuple(cache))
        new_request_state, r_t, R, P = env._step(action_index)
        total_cost += len(R) ** TARGET_FUNCTION
        t += 1
        if t % 1000 == 0:
            print("LFU:", t)
    avarage_cost = total_cost / t
    return avarage_cost

'''
def stable():
    request = np.zeros(K)
    for i in range(0, K):
        request[i] = random.randint(0, N)
    request = request.astype(int)
    cache = files
    random.shuffle(cache)
    cache = cache[0 : M]
    cache.sort()
    action_index = cob.index(tuple(cache))
    env = Environment(request, cache)
    total_cost = 0
    t = 0
    while t < MAX_ITERATION:
        new_request_state, r_t, R, P = env._step(action_index)
        total_cost += len(R) ** TARGET_FUNCTION
        t += 1
        if t % 1000 == 0:
            print("stable_t:", t)
    avarage_cost = total_cost / t
    return avarage_cost
'''

def MP():
    # open up a system environment
    request = np.zeros(K)
    for i in range(0, K):
        request[i] = random.randint(0, N)
    request = request.astype(int)
    cache = [n for n in range(1, M + 1)]
    env = Environment(request, cache)
    total_cost = 0
    t = 0
    new_request_state = request
    while t < MAX_ITERATION:
        R = list(set(new_request_state).difference(set(cache)))
        if len(R):
            R.sort()
            if R[0] == 0:
                del(R[0])
        # MP update
        action_index = cob.index(tuple(cache))
        new_request_state, r_t, R, P = env._step(action_index)
        total_cost += len(R) ** TARGET_FUNCTION
        t += 1
        if t % 1000 == 0:
            print("MP:", t)
    avarage_cost = total_cost / t
    return avarage_cost


def LMP():
    h = 100 #initial history length
    counter = np.zeros([K, h + 1])
    alpha = np.zeros([K, h])
    history = [[]for row in range(K)]
    possibility = np.zeros([K, N + 1])
    C_all = np.array([])
    C_all = C_all.astype(int)

    # open up a system environment
    request = np.zeros(K)
    for i in range(0, K):
        request[i] = random.randint(0, N)
    request = request.astype(int)
    cache = files
    random.shuffle(cache)
    cache = cache[0 : M]
    cache.sort()
    env = Environment(request, cache)
    total_cost = 0
    t = 0
    new_request_state = request
    while t < MAX_ITERATION:
        for i in range(K):
            history[i].append(new_request_state[i])
        R = list(set(new_request_state).difference(set(cache)))
        if len(R):
            R.sort()
            if R[0] == 0:
                del(R[0])
                
        no_push = cache.copy()
        no_push = np.append(no_push, R)
        no_push = np.unique(no_push)
        no_push = no_push.astype(int)

        '''            
        # LFU update
        if t >= h - 1:
            if len(history) > h:
                history = history[-h:]
                    
            for j in range(1, h + 1):
                for k in range(h - 1, j - 1, -1):
                    if history[k] == history[k - j]:
                        counter[j] += 1
            counter_per = counter / h
        '''
        
        if t >= h - 1:
            if len(history[0]) > h:
                for i in range(K):
                    history[i] = history[i][-h:]
                    
            for i in range(K):
                for j in range(1, h + 1):
                    for k in range(h - 1, j - 1, -1):
                        if history[i][k] == history[i][k - j]:
                            counter[i, j] += 1
            counter_per = counter / h

            # compute the possible push
            alpha = (1 - Q0) * counter_per

        '''        
        if t < h - 1:
            C = np.random.permutation(N) + 1
            C = C.astype(int)
            cache = np.sort(C[0 : M])
            
        else:
            for j in range(1, h + 1):
                possibility[history[h - j]] += alpha[i, j]
            for k in range(N + 1):
                possibility[k] += f[0, k]
            pos = np.argsort(-possibility[no_push])
            cache = no_push[pos[0 : M]]
        '''

        if t < h - 1:
            C = np.random.permutation(N) + 1
            C = C.astype(int)
            cache = np.sort(C[0 : M])
                
        else:
            for i in range(K):
                for j in range(1, h + 1):
                    possibility[i, history[i][h - j]] += alpha[i, j]
                for k in range(N + 1):
                    possibility[i, k] += f[0, k]
                pos = np.argsort(-possibility[i, no_push])
                if i == 0:
                    C_all = no_push[pos[0 : M]]
                else:
                    C_all = np.concatenate([C_all, no_push[pos[0 : M]]])
            count = Counter(C_all).most_common(M)
            for i in range(M):
                cache[i] = count[i][0]
        cache.sort()
        action_index = cob.index(tuple(cache))
        new_request_state, r_t, R, P = env._step(action_index)
        total_cost += len(R) ** TARGET_FUNCTION
        t += 1
        if t % 1000 == 0:
            print("LMP:", t)
    avarage_cost = total_cost / t
    return avarage_cost

def TLMP():
    h = 100 #initial history length
    counter = np.zeros([K, h + 1])
    alpha = np.zeros([K, h])
    history = [[]for row in range(K)]
    possibility = np.zeros([K, N + 1])
    C_all = np.zeros(K)
    C_all = C_all.astype(int)

    # open up a system environment
    request = np.zeros(K)
    for i in range(0, K):
        request[i] = random.randint(0, N)
    request = request.astype(int)
    cache = files
    random.shuffle(cache)
    cache = cache[0 : M]
    cache.sort()
    env = Environment(request, cache)
    total_cost = 0
    t = 0
    new_request_state = request
    while t < MAX_ITERATION:
        for i in range(K):
            history[i].append(new_request_state[i])
        R = list(set(new_request_state).difference(set(cache)))
        if len(R):
            R.sort()
            if R[0] == 0:
                 del(R[0])
                
        no_push = cache.copy()
        no_push = np.append(no_push, R)
        no_push = np.unique(no_push)
        no_push = no_push.astype(int)
            
        if t >= h - 1:
            if len(history[0]) > h:
                for i in range(K):
                    history[i] = history[i][-h:]
                    
            for i in range(K):
                for j in range(1, h + 1):
                    for k in range(h - 1, j - 1, -1):
                        if history[i][k] == history[i][k - j]:
                            counter[i, j] += 1
            counter_per = counter / h

            # compute the possible push
            alpha = (1 - Q0) * counter_per
        
        if t < h - 1:
            C = np.random.permutation(N) + 1
            C = C.astype(int)
            cache = np.sort(C[0 : M])
            
        else:                   
            if len(R) < math.sqrt(total_cost / (t + 1) / 2):
                pos1 = np.argmax(possibility, 1)
                P_file = Counter(pos1).most_common(1)[0][0]
                if P_file != 0:
                    all_files = np.append(no_push, P_file)
                else:
                    all_files = no_push
            else:
                all_files = no_push
            for i in range(K):
                for j in range(1, h + 1):
                    possibility[i, history[i][h - j]] += alpha[i, j]
                for k in range(N + 1):
                    possibility[i, k] += f[0, k]
                pos = np.argsort(-possibility[i, all_files])
                if i == 0:
                    C_all = all_files[pos[0 : M]]
                else:
                    C_all = np.concatenate([C_all, all_files[pos[0 : M]]])
            count = Counter(C_all).most_common(M)
            for i in range(M):
                cache[i] = count[i][0]        
        cache.sort()        
        action_index = cob.index(tuple(cache))
        new_request_state, r_t, R, P = env._step(action_index)
        total_cost += len(R) ** TARGET_FUNCTION
        t += 1
        if t % 1000 == 0:
            print("LMP:", t)
    avarage_cost = total_cost / t
    return avarage_cost

def main():
    #x = range(0, len(avarage_cost_RL))
    #plt.plot(x, loss)
    avarage_cost_RL, push_times, push_num = playGame()
    avarege_cost_LRU = LRU()
    avarege_cost_LFU = LFU()
    avarege_cost_MP = MP()
    avarege_cost_LMP = LMP()
    avarege_cost_TLMP = TLMP()
    print("RL:", avarage_cost_RL, "PUSH_TIMES:", push_times, "PUSH_NUM:", push_num)
    print("LRU:", avarege_cost_LRU)
    print("LFU:", avarege_cost_LFU)
    print("MP:", avarege_cost_MP)
    print("LMP:", avarege_cost_LMP)
    print("TLMP:", avarege_cost_TLMP)

if __name__ == "__main__":
    main()
