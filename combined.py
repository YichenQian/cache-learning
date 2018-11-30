from __future__ import print_function
from scipy.special import comb

import tensorflow as tf
import random
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
from itertools import combinations
from tensorflow.python.framework import ops
ops.reset_default_graph()

STEP_PER_ACTION = 1
TARGET_FUNCTION = 4
STABLE_CACHE = 0
MAX_ITERATION = 100000
NODE_NUM = 64
LEARNING_RATE = 1e-4

# Network parameters
K = 5  # The number of users
N = 6  # The number of total files
M = 3  # The cache size of edge servers
L = 2  # The cache size of users
C = np.zeros(M)  # The cache state of users
C_E = [n for n in range(1, L + 1)]  # The cache state of edge servers
ACTIONS = int(comb(N, M)) # number of valid actions
files = [n for n in range(1, N + 1)]
cob = list(combinations(files, M))
CNM = len(cob)
avarage_cost_all = 0
times = 1
target_fun = 2
NO_PUSH = 0
GAMMA = 0.01  # decay rate of past observations
TARGET_FUNCTION = 4


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
    
    def _user_request(self):
        # changes of user requests
        for i in range(0, K):
            tran_p = random.random()
            j = 0;
            while tran_p > f_sum[self.request_state[i], j]:
                j += 1
            self.request_state[i] = j
        return  self.request_state
    
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
        
        '''
        # changes of user requests
        for i in range(0, K):
            tran_p = random.random()
            j = 0;
            while tran_p > f_sum[self.request_state[i], j]:
                j += 1
            self.request_state[i] = j
        '''
        
        #return self.request_state, cost_0, R, P
        return cost_0, R, P
        
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

def trainNetwork(env, R_u, P_u, C_all, s, readout, sess):
    cache = env.cache_state
    request = env.request_state
    # mark
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

    # saving and loading networks
    saver = tf.train.Saver()
    sess.run(tf.initialize_all_variables())
    checkpoint = tf.train.get_checkpoint_state("saved_networks")
    if checkpoint and checkpoint.model_checkpoint_path:
        saver.restore(sess, checkpoint.model_checkpoint_path)
        print("Successfully loaded:", checkpoint.model_checkpoint_path)
    else:
        print("Could not find old network weights")
    
    
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

    RL = [len(R)]
    new_request_num = np.zeros(N + 1)
    for i in range(0, K):
        new_request_num[request[i]] += 1
    old_cache = cache.copy()
    cache = cob[action_index]
    # count the different cache nums
    cache_diff = len(set(cache).difference(set(old_cache)))
    
    cache_index = np.zeros(N)
    for i in cache:
        cache_index[i - 1] += 1
    #s_t1 = np.array(list(new_request_num) + list(cache_index) + RL)
    #total_cost += r_t * (min(N, M + K) ** TARGET_FUNCTION)
    #total_cost += r_t
    #avarage_cost = total_cost / (t + 1)

    # update the old values
    return R, P, cache, cache_diff, r_t

    
def playGame(env, R_u, P_u, C_all):
    sess = tf.InteractiveSession()
    s, readout = createNetwork()
    R, P, cache, cache_diff, r_t = trainNetwork(env, R_u, P_u, C_all, s, readout, sess)
    return R, P, cache, cache_diff, r_t 

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


def user_side(env, g, q_new, q_old, C_all):
    # Update the reacitve transmission
    R = np.zeros(K)
    R = R.astype(int)
    for i in range(K):
        if q_new[i] != 0 and q_new[i] not in C_all[i, :]:
            R[i] = q_new[i].copy()
    R_k = R.copy()
    R = np.unique(R)
    if R[0] == 0:
        np.delete(R, 0)
        
    C_old_all = C_all.copy()
    # Determine the per-uer push
    push_all = np.zeros(K)
    push_file_all = []
    P = np.array([])
    for i in range(K):
        possible_push = np.array([n for n in range(1, N + 1)])
        delete_index = C_old_all[i, :].copy()
        if R_k[i] != 0:
            delete_index = np.append(delete_index, R_k[i])
        np.delete(possible_push, (delete_index - 1).tolist())
        g_min = (R_k[i] != 0) ** target_fun / K + ((1 - R_k[i]) in C_E) ** target_fun
        push_num = 0
        A_k1 = q_new[i].copy()
        C_temp = C_old_all[i, :].copy()
        push_f = []
        for j in range(1, min(len(possible_push - 1), L) + 1):
            pos = np.argsort(g[possible_push - 1, A_k1])
            g_sort = np.sort(g[possible_push - 1, A_k1])
            sum_g = np.sum(g_sort[0 : j])
            push_file = possible_push[list(pos[0 : j])].copy()
            push_E = 0
            for k in range(j):
                if push_file[k] not in C_E:
                    push_E += 1
            pos1 = np.argsort(-g[C_old_all[i,:].astype(int) - 1, A_k1])
            g_sort = -np.sort(-g[possible_push - 1, A_k1])
            delete_g = sum(g_sort[0 : j])
            step_min = (R_k[i]!=0 + j) ** target_fun / K + sum_g - delete_g + (1 - (R_k[i] in C_E) + push_E) ** target_fun
            if step_min <= g_min:
                push_num = j
                g_min = step_min
                push_f = push_file.copy()
                C_temp = C_old_all[i, :].copy()
                C_temp = np.delete(C_temp, list(pos1[0 : j]))
                C_temp = np.concatenate([C_temp, push_f])
        push_file_all.append(push_f)
        push_all[i] = push_num
        P = np.concatenate([P, push_f])
        if len(C_all[i, :]) != len(C_temp):
            test = 2;
        C_all[i, :] = C_temp
        if C_all[i, 0] == C_all[i, 1]:
            test = 1;
    P = np.array(list(set(np.unique(P)).difference(set(R))))
    return R, P, C_all
    

def main():
    #x = range(0, len(avarage_cost_RL))
    #plt.plot(x, loss)
    #avarage_cost_RL, push_times, push_num = playGame()
    #avarege_cost_LRU = LRU()
    #avarege_cost_LFU = LFU()
    #avarage_cost_stable = stable()
    #print("RL:", avarage_cost_RL, "PUSH_TIMES:", push_times, "PUSH_NUM:", push_num)
    #print("LRU:", avarege_cost_LRU)
    #print("LFU:", avarege_cost_LFU)
    #print("stable:", avarage_cost_stable)
    t = 0
    total_cost = 0
    THRESHOLD = 1
    
    # open up a system environment
    request = np.zeros(K)
    for i in range(0, K):
        request[i] = random.randint(0, N)
    request = request.astype(int)
    # initialize cache at edge server side
    cache = files
    random.shuffle(cache)
    cache = cache[0 : M]
    env = Environment(request, cache)
    # initialize cache at user side
    C_all = np.zeros([K, L])  # user k file m is in C_all[k-1,m-1]
    C_all = C_all.astype(int)
    for i in range(K):
        C = np.random.permutation(N) + 1
        C = C.astype(int)
        C = np.sort(C[0 : L])
        C_all[i, :] = C.copy()
    
    while t <= MAX_ITERATION:
        diff = 100
        q_old = env.request_state
        q_new = env._user_request
        while diff >= THRESHOLD:
            diff = 0
            C_all_old = C_all
            cache_old = cache
            R_u, P_u, C_all = user_side(env, g, q_new, q_old, C_all)
            R, P, cache, cache_diff, r_t = playGame(env, R_u, P_u, C_all)
            diff += len(set(C_all).difference(set(C_all_old)))
            diff += len(set(cache).difference(set(cache_old)))
        total_cost += (len(P_u) + len(R_u)) ** target_fun
        total_cost += r_t
        avarage_cost = total_cost / (t + 1)
    
    return avarage_cost

if __name__ == "__main__":
    main()
