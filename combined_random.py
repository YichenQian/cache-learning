from __future__ import print_function

import random
import numpy as np
import pickle
#import matplotlib.pyplot as plt

GAMMA = 0.01 # decay rate of past observations
TARGET_FUNCTION = 2.0
MAX_ITERATION = 100000

# Network parameters
K = 5  # The number of users
N = 10  # The number of total files
M = 2  # The cache size of users
L = 3  # The cache size of edge servers
C = np.zeros(M)  # The cache state of users
#C_E = [n for n in range(1, L + 1)]  # The cache state of edge servers
files = [n for n in range(1, N + 1)]
avarage_cost_all_LRU = 0.0
avarage_cost_all_LFU = 0.0
avarage_cost_all_RAND = 0.0
times = 1
target_fun = 2.0
NO_PUSH = 0

# generate the transition probability
ga = 0.5
Q0 = 0.2
nn = 1.0
f =np.array([[0 for col in range(N + 1)] for row in range(N + 1)], dtype = float)
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

for time in range(0, times):
    q_new = np.zeros(K)  # req of user k is in q_new[k-1]
    for i in range(K):
        q_new[i] = random.randint(0, N)
    q_new = q_new.astype(int)
    C_all = np.zeros([K, M])  # user k file m is in C_all[k-1,m-1]
    C_all = C_all.astype(int)
    phi = np.zeros([1, 1, N, 1])
    for i in range(K):
        C = np.random.permutation(N) + 1
        C = C.astype(int)
        C = np.sort(C[0 : M])
        C_all[i, :] = C.copy()
    C_all_LRU = C_all.copy()
    C_all_LFU = C_all.copy()
    C_all_RAND = C_all.copy()
    C_old_all_LRU = C_all.copy()
    C_old_all_LFU = C_all.copy()
    C_old_all_RAND = C_all.copy()
    
    t = 0  # Time
    total_cost_LRU = 0.0
    total_cost_LFU = 0.0
    total_cost_RAND = 0.0
    avarage_cost_LRU = []
    avarage_cost_LFU = []
    avarage_cost_RAND = []
    frequency = np.zeros([K, M])
    
    # Begin the iteration
    while t <= MAX_ITERATION:
        # Update the edge server cache
        C_E = files.copy()
        random.shuffle(C_E)
        C_E = C_E[0 : L]
        
        # Update the user request
        r = np.random.randn(K)
        for i in range(K):
            tran_p = random.random()
            j = 0
            while tran_p > f_sum[q_new[i], j]:
                j += 1
            q_new[i] = j
        
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
        
        C_old_all_LRU = C_all_LRU.copy()
        C_old_all_LFU = C_all_LFU.copy()
        C_old_all_RAND = C_all_RAND.copy()
        
        # Determine the new cache
        for i in range(K):
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
        total_cost_LRU = total_cost_LRU + len(R_LRU) ** target_fun + len(R_E_LRU) ** target_fun
        total_cost_LFU = total_cost_LFU + len(R_LFU) ** target_fun + len(R_E_LFU) ** target_fun
        total_cost_RAND = total_cost_RAND + len(R_RAND) ** target_fun + len(R_E_RAND) ** target_fun
        avarage_cost_LRU.append(total_cost_LRU / t)
        avarage_cost_LFU.append(total_cost_LFU / t)
        avarage_cost_RAND.append(total_cost_RAND / t)    
    avarage_cost_all_LRU = avarage_cost_all_LRU + avarage_cost_LRU[-1]
    avarage_cost_all_LFU = avarage_cost_all_LFU + avarage_cost_LFU[-1]
    avarage_cost_all_RAND = avarage_cost_all_RAND + avarage_cost_RAND[-1]
avarage_cost_all_LRU = avarage_cost_all_LRU / times
avarage_cost_all_LFU = avarage_cost_all_LFU / times
avarage_cost_all_RAND = avarage_cost_all_RAND / times
print("LRU = {}".format(avarage_cost_all_LRU))
print("LFU = {}".format(avarage_cost_all_LFU))
print("RAND = {}".format(avarage_cost_all_RAND))

# save the result
#str1 = "training_data/g_{KK}_{NN}_{MM}_{LL}.txt".format(KK=K, NN=N, MM=M, LL=L)
#str2 = "training_data/Q_{KK}_{NN}_{MM}_{LL}.txt".format(KK=K, NN=N, MM=M, LL=L)
#pickle.dump(g_out,open(str1, 'wb'))
#pickle.dump(Q_out,open(str2, 'wb'))
#print ("write over")
