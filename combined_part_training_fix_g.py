#import sys
#if sys.getdefaultencoding()!='utf-8':
#    reload(sys)
#    sys.setdefaultencoding('utf-8')
from __future__ import print_function
from scipy.special import comb

import tensorflow as tf
import random
import numpy as np
import pickle
from collections import deque
from itertools import combinations
from tensorflow.python.framework import ops
ops.reset_default_graph()
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import argparse

STEP_PER_ACTION = 1
TARGET_FUNCTION = 2
STABLE_CACHE = 0
MAX_ITERATION = 30000000
NODE_NUM = 64
LEARNING_RATE = 1e-4
FINAL_EPSILON = 0.0001 # final value of epsilon
INITIAL_EPSILON = 0.1 # starting value of epsilon
REPLAY_MEMORY = 50000 # number of previous transitions to remember
BATCH = 32 # size of minibatch
OBSERVE = 20000. # timesteps to observe before training
EXPLORE = 3000000. # frames over which to anneal epsilon
GAMMA = 1 # decay rate of past observations
QUICK = 1
OBSERVE = OBSERVE / QUICK
EXPLORE = EXPLORE / QUICK
MAX_ITERATION = MAX_ITERATION / QUICK

# Network parameters
#K = 10  # The number of users
#N = 10  # The number of total files
parser = argparse.ArgumentParser(description='manual to this script')
parser.add_argument('--K', type=int, default = 0)
parser.add_argument('--N', type=int, default = 0)
args = parser.parse_args()
print(args.K)
print(args.N)
K = args.K
N = args.N
M = 3  #The cache size of edge servers
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
TARGET_FUNCTION = 2
# import g and Q
str1 = "training_data/g_{KK}_{NN}_{MM}_{LL}.txt".format(KK=K, NN=N, MM=L, LL=M)
str2 = "training_data/Q_{KK}_{NN}_{MM}_{LL}.txt".format(KK=K, NN=N, MM=L, LL=M)
g = pickle.load(open(str1, 'rb'))
Q_new = pickle.load(open(str2, 'rb'))
g_out = 0.0
Q_out = 0.0

#TEST
g = np.zeros([N, N + 1])
for i in range(N + 1):
    g[i % N, i] = -0.3

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
    
    def _step(self, cache_action, R_u, P_u):
        # compute the reactive transmission, push and cost
        R = list(set(R_u).difference(set(self.cache_state)))
        P_eu = list(set(P_u).difference(set(self.cache_state)))
        if len(R):
            R.sort()
            if R[0] == 0:
                del(R[0])
        self.old_cache_state = self.cache_state
        self.cache_state = list(cob[cache_action])
        add = list(set(self.cache_state).difference(set(self.old_cache_state)))
        P = list(set(add + P_eu).difference(set(R)))
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
    
    # open up a system environment
    request = np.zeros(K)
    for i in range(0, K):
        request[i] = random.randint(0, N)
    request = request.astype(int)
    q_new = request.copy()
    # initialization of edge server side
#    tf.InteractiveSession.close()
    sess = tf.InteractiveSession()
    s, readout = createNetwork()
    # cache at edge server side
    cache = files
    random.shuffle(cache)
    cache = cache[0 : M]
    env = Environment(request, cache)
    # define the cost function
    a = tf.placeholder("float", [None, ACTIONS])
    y = tf.placeholder("float", [None])
    readout_action = tf.reduce_sum(tf.multiply(readout, a), reduction_indices=1)
    #readout_action = tf.multiply(readout, a)
    cost = tf.reduce_mean(tf.square(y - readout_action))
    train_step = tf.train.AdamOptimizer(LEARNING_RATE).minimize(cost)
    # store the previous observations in replay memory
    D = deque()
    # epsilon
    epsilon = INITIAL_EPSILON
#       total_cost = 0
    LOSS_FUN = []
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
    s_0 = s_t
    '''
    # saving and loading networks
    saver = tf.train.Saver()
    checkpoint = tf.train.get_checkpoint_state("saved_networks")
    if checkpoint and checkpoint.model_checkpoint_path:
        saver.restore(sess, checkpoint.model_checkpoint_path)
        print("Successfully loaded:", checkpoint.model_checkpoint_path)
    else:
        print("Could not find old network weights")
    '''
    saver = tf.train.Saver()
    sess.run(tf.global_variables_initializer())
    
    # initialization of user side
    # cache at user side
    C_all = np.zeros([K, L])  # user k file m is in C_all[k-1,m-1]
    C_all = C_all.astype(int)
    for i in range(K):
        C = np.random.permutation(N) + 1
        C = C.astype(int)
        C = np.sort(C[0 : L])
        C_all[i, :] = C.copy()
    # Q and g
#    Q_new = np.zeros([2, 2, N, N + 1])
#    g = np.zeros([N, N + 1])  # g value
    # Count the number of requests
    push_file_all = []
    C_old_all = C_all.copy()
    
    while t <= MAX_ITERATION:
#        q_old = env.request_state
#        q_new = env._user_request()
        
        # user side
#        R_u, P_u, C_all = user_side(env, q_new, q_old, C_all, cache_old, t)    
        # Begin the iteration
        # Update the user request
        for i in range(K):
            tran_p = random.random()
            j = 0
            while tran_p > f_sum[q_new[i], j]:
                j += 1
            q_new[i] = j
        
        # Update the reacitve transmission
        R_u = np.zeros(K)
        R_u = R_u.astype(int)
        for i in range(K):
            if q_new[i] != 0 and (q_new[i] not in C_all[i, :]):
                R_u[i] = q_new[i].copy()
#        R_k = R_u.copy()
        R_u = np.unique(R_u)
        if R_u[0] == 0:
            R_u = np.delete(R_u, 0)
        R_e = list(set(R_u).difference(set(cache))) # R at edge server side
            
        C_old_all = C_all.copy()
        # Determine the per-uer push
        push_file_all = []
        P = np.array([])
        for i in range(K):
            possible_push = np.array([n for n in range(1, N + 1)])
            delete_index = C_old_all[i, :].copy()
            if len(R_u) != 0:
                delete_index = np.append(delete_index, R_u)
            temp_index = delete_index - 1
            possible_push = np.delete(possible_push, (temp_index).tolist())
            g_min = 1.0 / K * (len(R_u) ** target_fun + len(R_e) ** target_fun)
            push_num = 0
            A_k1 = q_new[i].copy()
            C_temp = C_old_all[i, :].copy()
            push_f = []
            # check the files in R if it can be better cache
            no_push = C_old_all[i, :].copy()
            no_push = np.append(no_push, R_u)
            no_push = np.unique(no_push)
            pos = np.argsort(g[no_push - 1, A_k1])
            g_sort = np.sort(g[no_push - 1, A_k1])
            sum_g = np.sum(g_sort[0 : L])
            pos1 = np.argsort(-g[C_old_all[i,:].astype(int) - 1, A_k1])
            g_sort1 = -np.sort(-g[C_old_all[i,:].astype(int) - 1, A_k1])
            delete_g = np.sum(g_sort1[0 : L])
            if delete_g > sum_g:
                g_min = g_min - delete_g + sum_g
                C_temp = np.delete(C_temp, list(pos1[0 : L]))
                C_temp = np.concatenate([C_temp, no_push[pos[0 : L]]])
            
            '''
            for j in range(1, min(len(possible_push - 1), L) + 1):
                pos = np.argsort(g[possible_push - 1, A_k1])
                g_sort = np.sort(g[possible_push - 1, A_k1])
                sum_g = np.sum(g_sort[0 : j])
                push_file = possible_push[list(pos[0 : j])].copy()
                push_E = 0
                for k in range(j):
                    if push_file[k] not in cache:
                        push_E += 1
                pos1 = np.argsort(-g[C_old_all[i,:].astype(int) - 1, A_k1])
                g_sort = -np.sort(-g[possible_push - 1, A_k1])
                delete_g = np.sum(g_sort[0 : j])
                #step_min = (R_k[i]!=0 + j) ** target_fun / K + sum_g - delete_g + (1 - (R_k[i] in C_E) + push_E) ** target_fun
                step_min =  1.0 / K * (len(R_u) + j) ** target_fun + sum_g - delete_g + 1.0 / K * (len(R_e) + push_E) ** target_fun

                if step_min <= g_min:
                    push_num = j
                    g_min = step_min
                    push_f = push_file.copy()
                    C_temp = C_old_all[i, :].copy()
                    C_temp = np.delete(C_temp, list(pos1[0 : j]))
                    C_temp = np.concatenate([C_temp, push_f])
            '''
            
            pos = np.argsort(g[possible_push - 1, A_k1])
            g_sort = np.sort(g[possible_push - 1, A_k1])
            sum_g = np.sum(g_sort[0])
            push_file = possible_push[pos[0]]
            push_E = 0
            if push_file not in cache:
                push_E += 1
            pos1 = np.argsort(-g[C_old_all[i,:].astype(int) - 1, A_k1])
            g_sort = -np.sort(-g[possible_push - 1, A_k1])
            delete_g = np.sum(g_sort[0])
            step_min =  1.0 / K * (len(R_u) + j) ** target_fun + sum_g - delete_g + 1.0 / K * (len(R_e) + push_E) ** target_fun
            if step_min <= g_min:
                g_min = step_min
                push_f = [push_file]
                C_temp = C_old_all[i, :].copy()
                C_temp = np.delete(C_temp, pos1[0])
                C_temp = np.concatenate([C_temp, push_f])
            P = push_f
            #if len(C_all[i, :]) != len(C_temp):
            #    test = 2;
            C_all[i, :] = C_temp
            #if C_all[i, 0] == C_all[i, 1]:
            #    test = 1;
        P_u = np.array(list(set(np.unique(P)).difference(set(R_u))))


        # edge server side            
#        R, P, cache, cache_diff, r_t = playGame(env, R_u, P_u, C_all, t)
        # Begin the iteration
        # start training    
        # choose an action epsilon greedily
        readout_t = readout.eval(feed_dict={s : [s_t]})[0]
        a_t = np.zeros([ACTIONS])
        action_index = 0
        #if t % STEP_PER_ACTION == 0:
        if STABLE_CACHE == 1:
            action_index = random.randrange(ACTIONS)
        a_t[action_index] = 1
        if STABLE_CACHE == 0:
            #action_index = np.argmin(readout_t)
            min_index = np.where(readout_t == np.min(readout_t))
            min_index = min_index[0]
            action_index = min_index[random.randint(0, len(min_index)-1)]
        a_t[action_index] = 1
        
        if epsilon > FINAL_EPSILON and t > OBSERVE:
                epsilon -= (INITIAL_EPSILON - FINAL_EPSILON) / EXPLORE
        # run the selected action and observe next state and reward
        r_t, R, P = env._step(action_index, R_u, P_u)
#        r_t += (len(P_u) + len(R_u)) ** TARGET_FUNCTION
    
        RL = [len(R)]
        new_request_num = np.zeros(N + 1)
        for i in range(0, K):
            new_request_num[q_new[i]] += 1
        cache = cob[action_index]
        # count the different cache nums
        
        cache_index = np.zeros(N)
        for i in cache:
            cache_index[i - 1] += 1
        s_t1 = np.array(list(new_request_num) + list(cache_index) + RL)
        #total_cost += r_t * (min(N, M + K) ** TARGET_FUNCTION)
        #total_cost += r_t
        #avarage_cost = total_cost / (t + 1)
    
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
    #       train_step.run(feed_dict = {y : y_batch, a : a_batch, s : s_j_batch})
            LOSS = sess.run(cost, feed_dict = {y : y_batch, a : a_batch, s : s_j_batch})
            if t % 1000 == 0:
                print("Times", t, "/ LOSS", LOSS)
                print("P", P, "R", R, "SYSTEM_STATE", s_t, "NEXT_STATE", s_t1)
            
            LOSS_FUN.append(LOSS)
            
            # save progress every (MAX_ITERATION / 10) iterations
            if t % (MAX_ITERATION / 10) == 0:
                str6 = "saved_networks_{KK}_{NN}_{MM}_{LL}_fixed_g/cached-dqn-".format(KK=K, NN=N, MM=M, LL=L)
                saver.save(sess, str6 , global_step = t)
    
            # save progress every 10000 iterations
    #        if t % 10000 == 0:
    #            saver.save(sess, 'saved_networks/' + 'cache' + '-dqn', global_step = t)
    #        if t % 1000 == 0:
    #            print("TIMESTEP", t, "/ AVARAGE_COST", avarage_cost, \
    #                  "/ ACTION", action_index, "/ REWARD", r_t)
            
            # update the old values
            s_t = s_t1
        
        t += 1
        total_cost += r_t + (len(P_u) + len(R_u)) ** TARGET_FUNCTION
#        total_cost += r_t
        avarage_cost = total_cost / t
        if t % 1000 == 1:
            print(avarage_cost)
    print(avarage_cost)
    sess.close()
    str5 = "simulation_result/cost_{KK}_{NN}_{MM}_{LL}_fixed_g.txt".format(KK=K, NN=N, MM=M, LL=L)
    pickle.dump(avarage_cost,open(str5, 'wb'))
    print ("write over")
    return avarage_cost

if __name__ == "__main__":
    main()
