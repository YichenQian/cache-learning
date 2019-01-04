from __future__ import print_function
import numpy as np
from numpy import *
from collections import Counter
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
#K = 5
#N = 10
print(K)
print(N)
M = 2  # The cache size of users
L = 3  # The cache size of edge servers
C = np.zeros(M)  # The cache state of users
C_E = [n for n in range(1, L + 1)]  # The cache state of edge servers
files = [n for n in range(1, N + 1)]
cob = list(combinations(files, M))
CNM = len(cob)
avarage_cost_all = 0.0
avarage_cost_all_LRU = 0.0
avarage_cost_all_LFU = 0.0
avarage_cost_all_RAND = 0.0
avarage_cost_all_MP = 0.0
avarage_cost_all_LMP = 0.0
avarage_cost_all_TLMP = 0.0
times = 1
target_fun = 2.0
NO_PUSH = 0
# import g
str1 = "training_data/g_{KK}_{NN}_{MM}_{LL}.txt".format(KK=K, NN=N, MM=M, LL=L)
g = pickle.load(open(str1, 'rb'))

'''
#TEST
g = np.zeros([N, N + 1])
for i in range(N + 1):
    g[i % N, i] = -0.2
'''

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

sigmap = np.sum(f[0, :] ** 2)

def cache_diff(old_cache_state, new_cache_state):
    add = list(set(new_cache_state).difference(set(old_cache_state)))
    delete = list(set(old_cache_state).difference(set(new_cache_state)))
    maintain = list(set(old_cache_state).intersection(set(new_cache_state)))
    return add, delete, maintain

for time in range(0, times):
    P_time = 0
    q_new = np.zeros(K)  # req of user k is in q_new[k-1]
    for i in range(K):
        q_new[i] = random.randint(0, N)
    q_new = q_new.astype(int)
    Q_new = np.zeros([2, 2, N, N + 1])
    Q_old = Q_new.copy()
    C_all = np.zeros([K, M])  # user k file m is in C_all[k-1,m-1]
    C_all = C_all.astype(int)
    phi = np.zeros([1, 1, N, 1])
    for i in range(K):
        C = np.random.permutation(N) + 1
        C = C.astype(int)
        C = np.sort(C[0 : M])
        C_all[i, :] = C.copy()
    C_old_all = C_all.copy()
    C_all_LRU = C_all.copy()
    C_all_LFU = C_all.copy()
    C_all_RAND = C_all.copy()
    C_all_MP = C_all.copy()
    C_all_LMP = C_all.copy()
    C_all_TLMP = C_all.copy()

    t = 0  # Time
    h = 100 #initial history length
    total_cost = 0.0  # Total bandwidth cost
    total_cost_LRU = 0.0
    total_cost_LFU = 0.0
    total_cost_RAND = 0.0
    total_cost_MP = 0.0
    total_cost_LMP = 0.0
    total_cost_TLMP = 0.0
    avarage_cost = []
    avarage_cost_LRU = []
    avarage_cost_LFU = []
    avarage_cost_RAND = []
    avarage_cost_MP = []
    avarage_cost_LMP = []
    avarage_cost_TLMP = []
    frequency = np.zeros([K, M])
    history = [[]for row in range(K)]
    history_temp = [[]for row in range(K)]
    counter = np.zeros([K, h + 1])
    counter_temp = np.zeros([K, h + 1])
    possibility = np.zeros([K, N + 1])
    
    # Count the number of requests
    req_times = np.zeros(N + 1)  # req_time of file n is in req_time[n], 0 represents no request
    alpha = np.zeros([K, h])
    beta = np.zeros(K)
    
    # Record the initial state
    A_k0 = random.randint(1, N)
    S_kf0 = 1
    dS_kf0 = 1
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
        
        # Random select the edge server cache
        C = np.random.permutation(N) + 1
        C = C.astype(int)
        C_E = np.sort(C[0 : L])

        '''
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
            
            '''
            # generate matrix A
            A = np.zeros([K, h, h])
            for i in range(K):
                for j in range(h):
                    for k in range(j, h):
                        if j == k:
                            A[i, j, j] = 1 - sigmap
                        else:
                            A[i, j, k] = counter[i, k - j] - sigmap
                            A[i, k, j] = counter[i, k - j] - sigmap
            B = np.zeros([K, h, 1])
            for i in range(K):
                for j in range(h):
                    B[i, j] = counter[i, j + 1]
            
            for i in range(K):
                AM = mat(A[i, :, :])
                BM = mat(B[i, :, :])
                alpha[i, :] = np.array((AM.I * BM).T)
                beta[i] = 1 - np.sum(alpha[i, :])
            '''
            
            # compute the possible push
            beta = Q0
            alpha = (1 - Q0) * counter_per

        # Update the reacitve transmission
        R = np.zeros(K)
        R = R.astype(int)
        for i in range(K):
            if q_new[i] != 0 and q_new[i] not in C_all[i, :]:
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
            
        # Update the reactive transmission for MP
        R_MP = np.zeros(K)
        R_MP = R_MP.astype(int)
        for i in range(K):
            if q_new[i] != 0 and q_new[i] not in C_all_MP[i, :]:
                R_MP[i] = q_new[i].copy()
        R_k_MP = R_MP.copy()
        R_MP = np.unique(R_MP)
        if R_MP[0] == 0:
            R_MP = np.delete(R_MP, 0)

        # Update the reactive transmission for LMP
        R_LMP = np.zeros(K)
        R_LMP = R_LMP.astype(int)
        for i in range(K):
            if q_new[i] != 0 and q_new[i] not in C_all_LMP[i, :]:
                R_LMP[i] = q_new[i].copy()
        R_k_LMP = R_LMP.copy()
        R_LMP = np.unique(R_LMP)
        if R_LMP[0] == 0:
            R_LMP = np.delete(R_LMP, 0)

        # Update the reactive transmission for TLMP
        R_TLMP = np.zeros(K)
        R_TLMP = R_TLMP.astype(int)
        for i in range(K):
            if q_new[i] != 0 and q_new[i] not in C_all_TLMP[i, :]:
                R_TLMP[i] = q_new[i].copy()
        R_k_TLMP = R_TLMP.copy()
        R_TLMP = np.unique(R_TLMP)
        if R_TLMP[0] == 0:
            R_TLMP = np.delete(R_TLMP, 0)
            
        # Determine the per-uer push
        push_all = np.zeros(K)
        push_file_all = []
        P = np.array([])
        for i in range(K):
            possible_push = np.array([n for n in range(1, N + 1)])
            delete_index = C_old_all[i, :].copy()
            if len(R) != 0:
                delete_index = np.append(delete_index, R)
            possible_push = np.delete(possible_push, (delete_index - 1).tolist())
            g_min = 1.0 / K * (len(R) ** target_fun + len(R_e) ** target_fun)
            #g_min = 1 / K * ((R_k[i] != 0) ** target_fun + (R_k[i] not in C_E) ** target_fun)
            push_num = 0
            A_k1 = q_new[i].copy()
            C_temp = C_old_all[i, :].copy()
            push_f = []
            # check the files in R if it can be better cache
            no_push = C_old_all[i, :].copy()
            no_push = np.append(no_push, R)
            no_push = np.unique(no_push)
            pos = np.argsort(g[no_push - 1, A_k1])
            g_sort = np.sort(g[no_push - 1, A_k1])
            sum_g = np.sum(g_sort[0 : M])
            pos1 = np.argsort(-g[C_old_all[i,:].astype(int) - 1, A_k1])
            g_sort1 = -np.sort(-g[C_old_all[i,:].astype(int) - 1, A_k1])
            delete_g = np.sum(g_sort1[0 : M])
            if delete_g > sum_g:
                g_min = g_min - delete_g + sum_g
                C_temp = np.delete(C_temp, list(pos1[0 : M]))
                C_temp = np.concatenate([C_temp, no_push[pos[0 : M]]])
            
            for j in range(1, min(len(possible_push - 1), M) + 1):
                pos = np.argsort(g[possible_push - 1, A_k1])
                g_sort = np.sort(g[possible_push - 1, A_k1])
                sum_g = np.sum(g_sort[0 : j])
                push_file = possible_push[list(pos[0 : j])].copy()
                push_E = 0
                for k in range(j):
                    if push_file[k] not in C_E:
                        push_E += 1
                pos1 = np.argsort(-g[C_old_all[i,:].astype(int) - 1, A_k1])
                g_sort1 = -np.sort(-g[C_old_all[i,:].astype(int) - 1, A_k1])
                delete_g = np.sum(g_sort1[0 : j])
                #step_min = 1 / K * (((R_k[i]!=0) + j) ** target_fun + sum_g - delete_g + ((R_k[i] not in C_E) + push_E) ** target_fun)
                step_min =  1.0 / K * (len(R) + j) ** target_fun + sum_g - delete_g + 1.0 / K * (len(R_e) + push_E) ** target_fun
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
            
            # MP
            req_sort = np.argsort(-req_times)
            if req_sort[0] == 0:
                C_all_MP[i, :] = req_sort[1 : 3]
            elif req_sort[1] == 0:
                C_all_MP[i, 0] = req_sort[0]
                C_all_MP[i, 1] = req_sort[2]
            else:
                C_all_MP[i, :] = req_sort[0 : 2]
            
            # LMP
            if t < h - 1:
                for i in range(K):
                    C = np.random.permutation(N) + 1
                    C = C.astype(int)
                    C = np.sort(C[0 : M])
                    C_all_LMP[i, :] = C.copy()
            else:
                for i in range(K):
                    for j in range(1, h + 1):
                        possibility[i, history[i][h - j]] += alpha[i, j]
                    for k in range(N + 1):
                        possibility[i, k] += f[0, k]
                    pos = np.argsort(-possibility[i, no_push])
                    C_all_LMP[i, :] = no_push[pos[0 : M]]
            
            # TLMP
            P_TLMP = []
            if t < h - 1:
                for i in range(K):
                    C = np.random.permutation(N) + 1
                    C = C.astype(int)
                    C = np.sort(C[0 : M])
                    C_all_TLMP[i, :] = C.copy()
            else:
                if len(R_TLMP) < math.sqrt(avarage_cost_all_LMP / 2):
                    pos1 = np.argmax(possibility, 1)
                    P_file = Counter(pos1).most_common(1)[0][0]
                    if P_file != 0:
                        all_files = np.append(no_push, P_file)
                        P_TLMP.append(P_file)
                        
                else:
                    all_files = no_push
                for i in range(K):
                    for j in range(1, h + 1):
                        possibility[i, history[i][h - j]] += alpha[i, j]
                    for k in range(N + 1):
                        possibility[i, k] += f[0, k]
                    pos = np.argsort(-possibility[i, all_files])
                    C_all_TLMP[i, :] = all_files[pos[0 : M]]
                
            
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
                
        # Compute the edge transmission and the edge push MP
        R_E_MP = []
        for i in range(len(R_MP)):
            if R_MP[i] not in C_E:
                R_E_MP.append(R_MP[i])
        
        # Compute the edge transmission and the edge push LMP
        R_E_LMP = []
        for i in range(len(R_LMP)):
            if R_LMP[i] not in C_E:
                R_E_LMP.append(R_LMP[i])
        
        # Compute the edge transmission and the edge push TLMP
        R_E_TLMP = []
        for i in range(len(R_TLMP)):
            if R_TLMP[i] not in C_E:
                R_E_TLMP.append(R_TLMP[i])
            
        P_E_TLMP = []
        for i in range(len(P_TLMP)):
            if P_TLMP[i] not in C_E:
                P_E_TLMP.append(P[i])
        
        # Compute the avarage cost
        t = t + 1
        if t % 1000 == 0:
            print(t)
        #print(C_all)
        total_cost = total_cost + (len(P) + len(R)) ** target_fun + (len(R_E) + len(P_E)) ** target_fun
        total_cost_LRU = total_cost_LRU + len(R_LRU) ** target_fun + len(R_E_LRU) ** target_fun
        total_cost_LFU = total_cost_LFU + len(R_LFU) ** target_fun + len(R_E_LFU) ** target_fun
        total_cost_RAND = total_cost_RAND + len(R_RAND) ** target_fun + len(R_E_RAND) ** target_fun
        total_cost_MP = total_cost_MP + len(R_MP) ** target_fun + len(R_E_MP) ** target_fun
        total_cost_LMP = total_cost_LMP + len(R_LMP) ** target_fun + len(R_E_LMP) ** target_fun
        total_cost_TLMP = total_cost_TLMP + (len(R_TLMP) + len(P_TLMP)) ** target_fun + (len(R_E_TLMP) + len(P_E_TLMP)) ** target_fun
        avarage_cost.append(total_cost / t)
        avarage_cost_LRU.append(total_cost_LRU / t)
        avarage_cost_LFU.append(total_cost_LFU / t)
        avarage_cost_RAND.append(total_cost_RAND / t)
        avarage_cost_MP.append(total_cost_MP / t)
        avarage_cost_LMP.append(total_cost_LMP / t)
        avarage_cost_TLMP.append(total_cost_TLMP / t)
    avarage_cost_all = avarage_cost_all + avarage_cost[-1]
    avarage_cost_all_LRU = avarage_cost_all_LRU + avarage_cost_LRU[-1]
    avarage_cost_all_LFU = avarage_cost_all_LFU + avarage_cost_LFU[-1]
    avarage_cost_all_RAND = avarage_cost_all_RAND + avarage_cost_RAND[-1]
    avarage_cost_all_MP = avarage_cost_all_MP + avarage_cost_MP[-1]
    avarage_cost_all_LMP = avarage_cost_all_LMP + avarage_cost_LMP[-1]
    avarage_cost_all_TLMP = avarage_cost_all_TLMP + avarage_cost_TLMP[-1]
avarage_cost_all = avarage_cost_all / times
avarage_cost_all_LRU = avarage_cost_all_LRU / times
avarage_cost_all_LFU = avarage_cost_all_LFU / times
avarage_cost_all_RAND = avarage_cost_all_RAND / times
avarage_cost_all_MP = avarage_cost_all_MP / times
avarage_cost_all_LMP = avarage_cost_all_LMP / times
avarage_cost_all_TLMP = avarage_cost_all_TLMP / times
print("ours = {}".format(avarage_cost_all))
print("LRU = {}".format(avarage_cost_all_LRU))
print("LFU = {}".format(avarage_cost_all_LFU))
print("RAND = {}".format(avarage_cost_all_RAND))
print("MP = {}".format(avarage_cost_all_MP))
print("LMP = {}".format(avarage_cost_all_LMP))
print("TLMP = {}".format(avarage_cost_all_TLMP))
print(P_time)
