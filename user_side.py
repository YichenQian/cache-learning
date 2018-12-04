from __future__ import print_function

import random
import numpy as np
import pickle
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
    C_old_all_LRU = C_all.copy()
    C_old_all_LFU = C_all.copy()
    C_old_all_RAND = C_all.copy()
    
    g = np.zeros([N, N + 1])  # g value
    count = np.zeros(N)
    
    t = 0  # Time
    total_cost = 0.0  # Total bandwidth cost
    total_cost_LRU = 0.0
    total_cost_LFU = 0.0
    total_cost_RAND = 0.0
    avarage_cost = []
    avarage_cost_LRU = []
    avarage_cost_LFU = []
    avarage_cost_RAND = []
    frequency = np.zeros([K, M])
    
    # Count the number of requests
    req_times = np.zeros(N + 1)  # req_time of file n is in req_time[n], 0 represents no request
    for i in range(K):
        req_times[q_new[i]] += 1
    
    # Record the initial state
    A_k0 = random.randint(1, N)
    S_kf0 = 1
    dS_kf0 = 1
    push_file_all = []
    
    # Begin the iteration
    while t <= MAX_ITERATION:
        Q_old = Q_new.copy()
        q_old = q_new.copy()
        # Update the user request
        r = np.random.randn(K)
        for i in range(K):
            tran_p = random.random()
            j = 0
            while tran_p > f_sum[q_new[i], j]:
                j += 1
            q_new[i] = j
        req_times[q_new[i]] += 1 # modified
        
        # Update the Q value and g
        if t != 0:
            Q_update = 0
            for i in range(K):
                req_old = q_old[i].copy()
                req_new = q_new[i].copy()
                
                # Find the cache action of last slot
                [add, delete, maintain] = cache_diff(C_old_all[i, :], C_all[i, :])
                add = np.array(add).astype(int)
                delete = np.array(delete).astype(int)
                maintain = np.array(maintain).astype(int)
                # Update the Q value of each cache action
                GAMA = np.zeros(N + 1)
                for j in range(N + 1):
                    if req_times[j] != 0:
                        GAMA[j] = 1.0 / np.sqrt(req_times[j])
                        #GAMA = 0.1 * np.ones(N + 1)
                
                # Compute phi(S_kf, f, A_k)
                P_n = len(push_file_all[i])
                r_c = np.zeros(N)
                for ff in range(1, N + 1):
                    r_c[ff - 1] = (1 - (req_old == ff) * (req_old in C_old_all[i, :]) * (req_old != 0))
                phi = 1.0 / K / N * ((r_c + P_n) ** target_fun + ((req_old not in C_E) + (P_n not in C_E)) ** target_fun)
                #phi = np.zeros([1, 1, N, 1])
                #for j in range(N):
                #    phi[1, 1, j, 1] = ph[j]
                
                # Add
                if len(add):
                    S_kf = 0  # 0 represent not cache, 1 represent cache
                    dS_kf = 1  # 0 represent maintain, 1 represent add
                    A_k = req_old  # no request = 0
                    S_kf1 = 1  # 0 represent not cache, 1 represent cache
                    A_k1 = req_new  # no request = 0
                    num = len(add)
                    Q_next = np.zeros(num)
                    for j in range(num):
                        c_a = (A_k1 == add[j]) * (1 - S_kf1)
                        Q_next[j] = Q_old[S_kf1, S_kf1 + c_a, add[j] - 1, A_k1].copy()
                    Q_new[S_kf, dS_kf, add - 1, A_k] = (1-GAMA[A_k]) * Q_old[S_kf, dS_kf, add - 1, A_k].copy() + GAMA[A_k]*(phi[add - 1] + Q_next - Q_old[S_kf0, dS_kf0, add - 1, A_k0].copy())
                    # Q_update = Q_update + sum(Q_new(S_kf, dS_kf, add, A_k) - Q_old(S_kf, dS_kf, add, A_k))
                
                # Delete
                if len(delete):
                    S_kf = 1  # 0 represent not cache, 1 represent cache
                    dS_kf = 0  # 0 represent delete, 1 represent maintain
                    A_k = req_old  # no request = 0
                    S_kf1 = 0  # 0 represent not cache, 1 represent cache
                    A_k1 = req_new  # no request = 0
                    num = len(delete)
                    Q_next = np.zeros(num)
                    for j in range(num):
                        c_a = (A_k1 == delete[j]) * (1 - S_kf1)
                        Q_next[j] = Q_old[S_kf1, S_kf1 + c_a, delete[j] - 1, A_k1].copy()
                    Q_new[S_kf, dS_kf, delete - 1, A_k] = (1-GAMA[A_k]) * Q_old[S_kf, dS_kf, delete - 1, A_k].copy() + GAMA[A_k]*(phi[delete - 1] + Q_next - Q_old[S_kf0, dS_kf0, delete - 1, A_k0].copy())
                    # Q_update = Q_update + sum(Q_new(S_kf, dS_kf, delete, A_k) - Q_old(S_kf, dS_kf, delete, A_k))
                
                # Maintain cache
                if len(maintain):
                    S_kf = 1  # 0 represent not cache, 1 represent cache
                    dS_kf = 1  # 0 represent delete, 1 represent maintain
                    A_k = req_old  # no request = 0
                    S_kf1 = 1  # 0 represent not cache, 1 represent cache
                    A_k1 = req_new  # no request = 0
                    num = len(maintain)
                    Q_next = np.zeros(num)
                    for j in range(num):
                        c_a = (A_k1 == maintain[j]) * (1 - S_kf1)
                        Q_next[j] = Q_old[S_kf1, S_kf1 + c_a, maintain[j] - 1, A_k1].copy()
                    Q_new[S_kf, dS_kf, maintain - 1, A_k] = (1-GAMA[A_k]) * Q_old[S_kf, dS_kf, maintain - 1, A_k].copy() + GAMA[A_k]*(phi[maintain - 1] + Q_next - Q_old[S_kf0, dS_kf0, maintain - 1, A_k0].copy())
                    # Q_update = Q_update + sum(Q_new(S_kf, dS_kf, maintain, A_k) - Q_old(S_kf, dS_kf, maintain, A_k))
                
                # Maintain not cache
                maintain_n = np.array(files)
                maintain_n = np.delete(maintain_n, add)
                maintain_n = np.delete(maintain_n, delete)
                maintain_n = np.delete(maintain_n, maintain)
                if len(maintain_n):
                    S_kf = 0  # 0 represent not cache, 1 represent cache
                    dS_kf = 1  # 0 represent maintain, 1 represent add
                    A_k = req_old  # no request = 0
                    S_kf1 = 1  # 0 represent not cache, 1 represent cache
                    A_k1 = req_new  # no request = 0
                    num = len(maintain_n)
                    Q_next = np.zeros(num)
                    for j in range(num):
                        c_a = (A_k1 == maintain_n[j]) * (1 - S_kf1)
                        Q_next[j] = Q_old[S_kf1, S_kf1 + c_a, maintain_n[j] - 1, A_k1].copy()
                    Q_new[S_kf, dS_kf, maintain_n - 1, A_k] = (1-GAMA[A_k]) * Q_old[S_kf, dS_kf, maintain_n - 1, A_k].copy() + GAMA[A_k]*(phi[maintain_n - 1] + Q_next - Q_old[S_kf0, dS_kf0, maintain_n - 1, A_k0].copy())
                    # Q_update = Q_update + sum(Q_new(S_kf, dS_kf, add, A_k) - Q_old(S_kf, dS_kf, add, A_k))
                
                # Update g
                #print(A_k)
                for j in range(N):
                    S_kf = (j + 1 in C_old_all[i, :]) + 0
                    #g[j, A_k] = min(Q_new[S_kf0, :, j, A_k0]) + Q_new[S_kf, S_kf, j, A_k].copy() - phi[j]
                    g[j, A_k] = Q_new[1, 1, j, A_k].copy() + Q_new[0, 0, j, A_k].copy() - phi[j]
                
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
        
        C_old_all = C_all.copy()
        C_old_all_LRU = C_all_LRU.copy()
        C_old_all_LFU = C_all_LFU.copy()
        C_old_all_RAND = C_all_RAND.copy()
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
