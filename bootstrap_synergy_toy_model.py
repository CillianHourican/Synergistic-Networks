# -*- coding: utf-8 -*-
"""
Created on Fri Aug 19 12:36:27 2022

@author: Cillian
"""
import multiprocess
import numpy as np
import time

import pandas as pd
from JointProbabilityMatrix import JointProbabilityMatrix


from npeet import entropy_estimators as ee
import itertools
from measures import synergistic_information
from measures import synergistic_information_naive
import dit

#from npeet import entropy_estimators as ee
def compute_o_info(data):
    '''
    data is a pandas dataframe [num_observations, num_variables]

    Parameters
    ----------
    data : pandas dataframs
        Shape [num_observations, num_variables]

    Returns
    -------
    numpy.float64

    '''
    o_info = (len(data.columns) - 2)*ee.entropyd(data)
    for j,_ in enumerate(data.columns):
        #o_info += ee.entropyd(data.loc[:,data.columns == j])
        o_info += ee.entropyd(data.loc[:,data.columns == _])
        o_info -= ee.entropyd(data.loc[:,data.columns != _])
    
    return(o_info)

def calculate_bootstrap_estimates(N,df,block_id = 0, sema = None):
    # Sample size of 50    
    Num_rows = df.shape[0]
    N_size = 100#int(round(min(50, Num_rows*0.4),0))
    
    cols = ['$S_1$', '$S_2$', '$S_3$', '$S_4$', '$S_5$','$S_6$', '$D_1$', '$D_2$', '$D_3$']
    a_cols = a_cols = [4, 0, 1, 5, 7,2,6,8,3] 
    
    #N = 3 # Size of groups
    undirected_groups = list(itertools.combinations(cols, N))
    #directed_groups = list(itertools.permutations(cols, N))
    

    bootstrap_sample = pd.DataFrame(df.values[np.random.randint(Num_rows, size=N_size)], columns=cols)
    
    dict_simulation_info = []
    
    for triplet in undirected_groups:
        A = []
        for indx,node in enumerate(triplet):
            # Map triplet back to variables in the toy model
            A.append(a_cols[cols.index(node)])
    
        o_info = compute_o_info(bootstrap_sample[bootstrap_sample.columns[A]])
        
        data_subset = bootstrap_sample.iloc[:, A]
        data_subset_list = data_subset.values.tolist()  # dataframe to list of list.
        pdf = JointProbabilityMatrix(2, 2)
        pdf.estimate_from_data(data_subset_list)
        d = dit.Distribution.from_ndarray(pdf.joint_probabilities.joint_probabilities)
        II = dit.multivariate.interaction_information(d)
    
        dict_simulation_info.append({'triplet': triplet, '-O-info': o_info, 'interaction_information':II})

    df_simulation_info = pd.DataFrame(dict_simulation_info)
    
    df_simulation_info.to_csv("simulations/bootstrap_B100_undirected_synergy_groups_"+str(N)+"_block_"+str(block_id)+".csv")
    
    # directed_groups = list(itertools.permutations(cols, N))
    
    # # Remove duplicates
    # for groupA in directed_groups:
    #     for groupB in directed_groups:
    #         if (groupA != groupB) and (groupB[0] == groupA[0]):
    #             if (set(groupA) == set(groupB) ):
    #                 directed_groups.remove(groupB)
    
    # #synergistic_information_naive(a,[2], [0,5])
    # dict_simulation_info = []
    
    # # Create network model from the bootstrapped samples
    # data_subset_list = bootstrap_sample.values.tolist()  # dataframe to list of list.
    # pdf = JointProbabilityMatrix(2, 2)
    # pdf.estimate_from_data(data_subset_list)
    
    # for triplet in directed_groups:
    #     A = []
    #     for indx,node in enumerate(triplet):
    #         # Map triplet back to variables in the toy model
    #         A.append(a_cols[cols.index(node)])

    #     syn_1_dict = np.sum(synergistic_information(pdf, [A[0]], A[1:])['I(Y;S)'])
    #     #syn_2_dict = np.sum(synergistic_information(a, [B], [A, C])['I(Y;S)'])
    #     #syn_3_dict = np.sum(synergistic_information(a, [C], [B, A])['I(Y;S)'])
    #     #syn_12_0 = np.sum(syn_12_0_dict['I(Y;S)'])
    #     syn_naive1 = synergistic_information_naive(pdf, [A[0]], A[1:])
    #     #syn_naive2 = synergistic_information_naive(a, [B], [A, C])
    #     #syn_naive3 = synergistic_information_naive(a, [C], [B, A])
    
    #     dict_simulation_info.append({'triplet': triplet, 'SRV': syn_1_dict,'WMS':syn_naive1})

    # df_simulation_info = pd.DataFrame(dict_simulation_info)
    
    # df_simulation_info.to_csv("simulations/bootstrap_directed_synergy_groups_"+str(N)+"_block_"+str(block_id)+".csv")
    
    if sema:
        sema.release()


def run_bootstrap():
    start = time.time()
    df = pd.read_csv('simulations/V2_toy_model_OR_rearranged_5.csv')
    df = df.loc[:, df.columns != "Unnamed: 0"]

    # Number of Bootstrap Samples
    #N_blocks = 100
    
    jobs = []
    
    sema = multiprocess.Semaphore(int(multiprocess.cpu_count()-2))
    print("Num cores:", int(multiprocess.cpu_count()) )
    
    for N in range(3,9):
        for sim in range(1000):
            sema.acquire()
            p = multiprocess.Process(target=calculate_bootstrap_estimates, args=(N,df,sim,sema)  )
            jobs.append(p)
            p.start()  
    
    # N = 4       
    # for sim in [23,50,52]:#, ##range(1):
    #     sema.acquire()
    #     p = multiprocess.Process(target=calculate_bootstrap_estimates, args=(N,df,sim,sema)  )
    #     jobs.append(p)
    #     p.start()  
        
    print("Number of jobs: " + str(len(jobs)))    
    count = 0
    for p in jobs:
        count += 1
        p.join()
        print("jobs done...", count)
        

    print("All jobs finished")
    print("time taken:", time.time() - start)
    
    
if __name__ == '__main__':
    #run_sims()
    run_bootstrap()
    