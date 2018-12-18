from __future__ import print_function
from numpy import *
import numpy as np
import random
import pickle
import math
#import matplotlib.pyplot as plt
from itertools import combinations
import argparse

GAMMA = 0.01 # decay rate of past observations
TARGET_FUNCTION = 2.0
MAX_ITERATION = 100000

# Network parameters
#K = 15  # The number of users
#N = 10  # The number of total files
parser = argparse.ArgumentParser(description='manual to this script')
parser.add_argument('--K', type=int, default = 0)
parser.add_argument('--N', type=int, default = 0)
args = parser.parse_args()
K = args.K
N = args.N
print(K)
print(N)
M = 2  # The cache size of users
L = 3  # The cache size of edge servers
C = np.zeros(M)  # The cache state of users
C_E = [n for n in range(1, N + 1)]  # The cache state of edge servers
files = [n for n in range(1, N + 1)]
cob = list(combinations(files, M))
CNM = len(cob)
avarage_cost_all = 0.0
avarage_cost_all_LRU = 0.0
avarage_cost_all_LFU = 0.0
avarage_cost_all_RAND = 0.0
times = 1
target_fun = 2.0
NO_PUSH = 0
g_out = 0.0
Q_out = 0.0

# generate the transition probability
ga = 0.5
Q0 = 0.2
nn = 1.0
f = np.array([[0 for col in range(N + 1)] for row in range(N + 1)], dtype = float)
f_sum = np.array([[0 for col in range(N + 1)] for row in range(N + 1)], dtype = float)
sum_pro = 0
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
            f[i, j] = (1 - Q0) / nn
for i in range(0, N + 1):
    for j in range(0, N + 1):
        f_sum[i, j] = f[i, 0 : j + 1].sum()
sigmap = sum(f[0,:] ** 2)

def cache_diff(old_cache_state, new_cache_state):
    add = list(set(new_cache_state).difference(set(old_cache_state)))
    delete = list(set(old_cache_state).difference(set(new_cache_state)))
    maintain = list(set(old_cache_state).intersection(set(new_cache_state)))
    return add, delete, maintain

for time in range(0, times):
    P_time = 0
    
    C_all = np.zeros([K, M])  # user k file m is in C_all[k-1,m-1]
    C_all = C_all.astype(int)
    phi = np.zeros([1, 1, N, 1])
    for i in range(K):
        C = np.random.permutation(N) + 1
        C = C.astype(int)
        C = np.sort(C[0 : M])
        C_all[i, :] = C.copy()
    C_all_MP = C_all.copy()
    C_all_LMP = C_all.copy()
    C_all_TLMP = C_all.copy()
#    C_old_all_MP = C_all.copy()
#    C_old_all_LMP = C_all.copy()
#    C_old_all_TLMP = C_all.copy()
    
    t = 0  # Time
    h = 50 #initial history length
    total_cost_MP = 0.0
    total_cost_LMP = 0.0
    total_cost_TLMP = 0.0
    avarage_cost = []
    avarage_cost_MP = []
    avarage_cost_LMP = []
    avarage_cost_TLMP = []
    histroy = [[]for row in range(K)]
    history_temp = [[]for row in range(K)]
    counter = np.zeros([K, h + 1])
    counter_temp = np.zeros([K, h + 1])    
    # Count the number of requests
    req_times = np.zeros(N + 1)  # req_time of file n is in req_time[n], 0 represents no request
    
    # Record the initial state
    push_file_all = []
    
    # Begin the iteration
    while t <= MAX_ITERATION:
        # Update the user request
        r = np.random.randn(K)
        for i in range(K):
            tran_p = random.random()
            j = 0
            while tran_p > f_sum[q_new[i], j]:
                j += 1
            q_new[i] = j
            req_times[q_new[i]] += 1
            history[i].append(q_new[i])        

        if t < h - 1:
            for i in range(K):
                C = np.random.permutation(N) + 1
                C = C.astype(int)
                C = np.sort(C[0 : M])
                C_all_LMP[i, :] = C.copy()

        elif t == h - 1:
            for i in range(K):
                for j in range(1, h + 1):
                    for k in range(h - 1, j - 1, -1):
                        if history[i][k] == history[i][k - j]:
                            counter[i, j] += 1
            average_counter = np.sum(counter, 1) / h
            history_temp = history
            for i in range(K):
                random.shuffle(history_temp[i])
            for i in range(K):
                for j in range(1, h + 1):
                    for k in range(h - 1, j - 1, -1):
                        if history_temp[i][k] == history_temp[i][k - j]:
                            counter_temp[i, j] += 1
            h_count = np.zeros(K)
            for i in range(K):
                for j in range(1, h + 1):
                    if counter_temp[i, j] > average_counter[i, j]:
                        h_count[i] += 1
            h = np.sum(h_count) / K
            
        else:
            if len(history[0]) > h:
                for i in range(K):
                    history[i] = history[i][-h:]
            
            # generate matrix A
            A = np.zeros([K, h, h])
            for i in range(K):
                for j in range(h):
                    for k in range(j, h):
                        if j == k:
                            A[i, j, j] = 1 - sigmap
                        else:
                            A[i, j, k] = counter[i, k - j] 
                            A[i, k, j] = 

        # Update the reacitve transmission
        R = np.zeros(K)
        R = R.astype(int)
        for i in range(K):
            if q_new[i] != 0 and (q_new[i] not in C_all[i, :]):
                R[i] = q_new[i].copy()
        R_k = R.copy()
        R = np.unique(R)
        if R[0] == 0:
            R = np.delete(R, 0)
        R_e = list(set(R).difference(set(C_E))) # R at edge server side
        
        # Update the reacitve transmission for LRU
        R_LRU = np.zeros(K)
        R_LRU = R_LRU.astype(int)
        for i in range(K):
            if q_new[i] != 0 and q_new[i] not in C_all_LRU[i, :]:
                R_LRU[i] = q_new[i].copy()
        R_k_LRU = R_LRU.copy()
        R_LRU = np.unique(R_LRU)
        if R_LRU[0] == 0:
            R_LRU = np.delete(R_LRU, 0)
        
        # Update the reacitve transmission for LFU
        R_LFU = np.zeros(K)
        R_LFU = R_LFU.astype(int)
        for i in range(K):
            if q_new[i] != 0 and q_new[i] not in C_all_LFU[i, :]:
                R_LFU[i] = q_new[i].copy()
        R_k_LFU = R_LFU.copy()
        R_LFU = np.unique(R_LFU)
        if R_LFU[0] == 0:
            R_LFU = np.delete(R_LFU, 0)
        
        # Update the reacitve transmission for RANDOM
        R_RAND = np.zeros(K)
        R_RAND = R_RAND.astype(int)
        for i in range(K):
            if q_new[i] != 0 and q_new[i] not in C_all_RAND[i, :]:
                R_RAND[i] = q_new[i].copy()
        R_k_RAND = R_RAND.copy()
        R_RAND = np.unique(R_RAND)
        if R_RAND[0] == 0:
            R_RAND = np.delete(R_RAND, 0)
        
        C_old_all = C_all.copy()
        C_old_all_LRU = C_all_LRU.copy()
        C_old_all_LFU = C_all_LFU.copy()
        C_old_all_RAND = C_all_RAND.copy()
 
    
            #if len(C_all[i, :]) != len(C_temp):
            #    test = 2
            C_all[i, :] = C_temp
            #if C_all[i, 0] == C_all[i, 1]:
            #    test = 1
            # LRU
            if R_k_LRU[i] != 0:
                C_all_LRU[i, :] = np.concatenate([[R_k_LRU[i]], C_all_LRU[i, 0 : -1]])
            
            # LFU
            if q_new[i] in C_all_LFU[i, :]:
                frequency[i, np.argwhere(C_all_LFU[i, :] == q_new[i])[0,0]] += 1
            f_temp = frequency[i, :]
            if R_k_LFU[i] != 0:
                index = f_temp.argmin()
                frequency[i, index] = 1
                C_all_LFU[i, index] = R_k_LFU[i]
            
            # RANDOM
            C = np.random.permutation(N) + 1
            C = C.astype(int)
            C = np.sort(C[0 : M])
            C_all_RAND[i, :] = C.copy()
            
        P = np.array(list(set(np.unique(P)).difference(set(R))))
        if len(P) != 0:
            P_time += 1
        
        # Compute the edge transmission and the edge push
        R_E = []
        for i in range(len(R)):
            if R[i] not in C_E:
                R_E.append(R[i])
        P_E = []
        for i in range(len(P)):
            if P[i] not in C_E:
                P_E.append(P[i])
                
        # Compute the edge transmission and the edge push LRU
        R_E_LRU = []
        for i in range(len(R_LRU)):
            if R_LRU[i] not in C_E:
                R_E_LRU.append(R_LRU[i])
                
        # Compute the edge transmission and the edge push LFU
        R_E_LFU = []
        for i in range(len(R_LFU)):
            if R_LFU[i] not in C_E:
                R_E_LFU.append(R_LFU[i])
        
        # Compute the edge transmission and the edge push RANDOM
        R_E_RAND = []
        for i in range(len(R_RAND)):
            if R_RAND[i] not in C_E:
                R_E_RAND.append(R_RAND[i])
        
        # Compute the avarage cost
        t = t + 1
        if t % 1000 == 0:
            print(t)
        #print(C_all)
        total_cost = total_cost + (len(P) + len(R)) ** target_fun + (len(R_E) + len(P_E)) ** target_fun
        total_cost_LRU = total_cost_LRU + len(R_LRU) ** target_fun + len(R_E_LRU) ** target_fun
        total_cost_LFU = total_cost_LFU + len(R_LFU) ** target_fun + len(R_E_LFU) ** target_fun
        total_cost_RAND = total_cost_RAND + len(R_RAND) ** target_fun + len(R_E_RAND) ** target_fun
        avarage_cost.append(total_cost / t)
        avarage_cost_LRU.append(total_cost_LRU / t)
        avarage_cost_LFU.append(total_cost_LFU / t)
        avarage_cost_RAND.append(total_cost_RAND / t)    
    g_out = g
    Q_out = Q_new
    avarage_cost_all = avarage_cost_all + avarage_cost[-1]
    avarage_cost_all_LRU = avarage_cost_all_LRU + avarage_cost_LRU[-1]
    avarage_cost_all_LFU = avarage_cost_all_LFU + avarage_cost_LFU[-1]
    avarage_cost_all_RAND = avarage_cost_all_RAND + avarage_cost_RAND[-1]
avarage_cost_all = avarage_cost_all / times
avarage_cost_all_LRU = avarage_cost_all_LRU / times
avarage_cost_all_LFU = avarage_cost_all_LFU / times
avarage_cost_all_RAND = avarage_cost_all_RAND / times
print("ours = {}".format(avarage_cost_all))
print("LRU = {}".format(avarage_cost_all_LRU))
print("LFU = {}".format(avarage_cost_all_LFU))
print("RAND = {}".format(avarage_cost_all_RAND))
print(P_time)

# save g to g.txt
str1 = "training_data/g_{KK}_{NN}_{MM}_{LL}.txt".format(KK=K, NN=N, MM=M, LL=L)
str2 = "training_data/Q_{KK}_{NN}_{MM}_{LL}.txt".format(KK=K, NN=N, MM=M, LL=L)
pickle.dump(g_out,open(str1, 'wb'))
pickle.dump(Q_out,open(str2, 'wb'))
print ("write over")
