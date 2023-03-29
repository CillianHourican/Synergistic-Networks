# -*- coding: utf-8 -*-
"""
Created on Thu Mar 10 18:28:15 2022

@author: Cillian
"""


#############

import pandas as pd
from JointProbabilityMatrix import JointProbabilityMatrix
import toy_functions as tf
import numpy as np

# num_var = 1
# num_val = 2
# num_sample = 100000
# mi_factor = 0.9
# diagnosis_factor = 0.9

# # [0] Create one variable
# a = JointProbabilityMatrix(1, num_val,joint_probs="uniform")
# print(len(a))

# # # Add symptoms with correlation to symptom
# # tf.append_variables_with_target_corr(a, 1, 0.5, relevant_variables =[0])
# # print(len(a))


# #[1][2] Append independent variables
# in_times = 2
# for jj in range(in_times):
#     b = JointProbabilityMatrix(1, num_val,joint_probs="uniform")
#     tf.append_independent_variables(a, b)
#     print(len(a))   
    

# ####
    
# # [3] Add symptoms with mutual information to diseases
# tf.append_variables_with_target_mi(a, 1, a.entropy([0]) * mi_factor, relevant_variables =[0])
# print(len(a))
    
# # [4] Append synergictic variables 0,1-> 4
# try:
#     tf.append_synergistic_variables(a, 1, subject_variables=[0,1])
# except UserWarning as e:
#     assert 'minimize() failed'
# print(len(a))   

# # [5] Add symptoms with mutual information to diseases
# tf.append_variables_with_target_mi(a, 1, a.entropy([3,4]) * diagnosis_factor, relevant_variables =[3,4])
# print(len(a))

# # [6] 1->6
# tf.append_variables_with_target_mi(a, 1, a.entropy([1]) * mi_factor, relevant_variables =[1])
# print(len(a))

# # [7]
# tf.append_variables_with_target_mi(a, 1, a.entropy([6]) * diagnosis_factor, relevant_variables =[6])
# print(len(a))


# # [8] 
# attempt = 0
# while len(a) < 9 and attempt < 30:
#     attempt += 1
#     # Append two synergictic variables 1,6 -> 8
#     try:
#         tf.append_synergistic_variables(a, 1, subject_variables=[1,6])
#         print(len(a))
#     except:
#         print("Failed to add synergistic variable. Trying again..")
    

# # except UserWarning as e:
# #     assert 'minimize() failed'
# # print(len(a))
    
# # [9]    
# tf.append_variables_with_target_mi(a, 1, a.entropy([2,8]) * diagnosis_factor, relevant_variables =[2,8])
# print(len(a))



# # Append two synergictic variables 6,2 -> 7
# # ToDo: Want to add synergistic edges but not new variables. Is there a function for this?
# try:
#     tf.append_synergistic_variables(a, 2, subject_variables=[4])
# except UserWarning as e:
#     assert 'minimize() failed'
# print(len(a))

# # ToDo: Add additional MI links between (already created) variables



# # Append independent variables
# in_times = 3
# for jj in range(in_times):
#     b = JointProbabilityMatrix(num_var, num_val)
#     tf.append_independent_variables(a, b)

# print(len(a))

# # Append synergistic variables
# try:
#     tf.append_synergistic_variables(a, 1, subject_variables=[1, 4])
# except UserWarning as e:
#     assert 'minimize() failed'
# print(len(a))

# Save network
import pickle
import itertools

'''
version_num = 0
with open('toy_model_V'+str(version_num)+'.pickle', 'wb') as handle:
    pickle.dump(a, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open('toy_model.pickle', 'rb') as handle:
    b = pickle.load(handle)

print(a == b)

b.mutual_information([5],[9])



tf.append_variables_using_state_transitions_table(a,[[0,0,0,0],[1,1,1,1]])
tf.append_variables_using_state_transitions_table(a,generate_XOR_transitions_table(a,dependent_vars=[0,1]))
'''
#from numba import jit

def generate_N_XOR_transitions_table(graph, dependent_vars):
    # N is the number of inputs in the XOR gate
    # The output should be 1 if the number of inputs is ODD
    #ToDo: allow for more then two dependent variables
    
    # list of dependent vars
    n = len(graph)
    
    # Generate a list of lists
    lst = [list(i) for i in itertools.product([0, 1], repeat=n)]
    
    # Generate XOR
    for _,i in enumerate(lst):
        total = 0
        for var in dependent_vars:
            total += i[var]
        #if total ==0:
        #    lst[_].append(0)
        if total%2==1:
            lst[_].append(1)
        else:
           lst[_].append(0) 
          
    return(lst)


def generate_XOR_transitions_table(graph, dependent_vars):
    #ToDo: allow for more then two dependent variables
    
    # list of dependent vars
    n = len(graph)
    
    # Generate a list of lists
    lst = [list(i) for i in itertools.product([0, 1], repeat=n)]
    
    # Generate XOR
    for _,i in enumerate(lst):
        if i[dependent_vars[0]] != i[dependent_vars[1]]:
            lst[_].append(1)
        else:
           lst[_].append(0) 
          
    return(lst)


def generate_OR_transitions_table(graph, dependent_vars):
    #ToDo: allow for more then two dependent variables
    
    # list of dependent vars
    n = len(graph)
    
    # Generate a list of lists
    lst = [list(i) for i in itertools.product([0, 1], repeat=n)]
    
    # Generate XOR
    for _,i in enumerate(lst):
        if i[dependent_vars[0]] == i[dependent_vars[1]]==0:
            lst[_].append(0)
        else:
           lst[_].append(1) 
          
    return(lst)

#from numba import jit
def generate_OR_transitions_table2(graph, dependent_vars):
    #ToDo: allow for more then two dependent variables
    
    # list of dependent vars
    n = len(graph)
    
    # Generate a list of lists
    lst = [list(i) for i in itertools.product([0, 1], repeat=n)]
    
    # Generate XOR
    for _,i in enumerate(lst):
        if i[dependent_vars[0]] == i[dependent_vars[1]]==0:
            lst[_].append(0)
        else:
           lst[_].append(1) 
          
    return(lst)


def generate_AND_transitions_table(graph, dependent_vars):
    #ToDo: allow for more then two dependent variables
    
    # list of dependent vars
    n = len(graph)
    
    # Generate a list of lists
    lst = [list(i) for i in itertools.product([0, 1], repeat=n)]
    
    # Generate XOR
    for _,i in enumerate(lst):
        if i[dependent_vars[0]] == i[dependent_vars[1]]==1:
            lst[_].append(1)
            print("appended")
        else:
           lst[_].append(0) 
          
    return(lst)
    
    

# # Generate transition table with independent vars
# import itertools
# n = 3

# # this is a list of tuples
# lst = list(itertools.product([0, 1], repeat=n))
# lst = [list(i) for i in itertools.product([0, 1], repeat=n)]

# dependent_vars = [0,1]

# for _,i in enumerate(lst):
#     if i[dependent_vars[0]] != i[dependent_vars[1]]:
#         lst[_].append(1)
#     else:
#        lst[_].append(0) 

# # convert to list of lists

# def generate_outputs(dependent_vars):
    

# # Generate XOR
# def XOR (a, b): 
#     if a != b: 
#         return 1
#     else: 
#         return 0 
    
''' 
# dd = a.conditional_probability_distribution([0],[1])
# toy_data = dd.generate_samples(10)
# toy_data

# ee =dd.joint_probabilities.joint_probabilities
# ee.sum(axis = (0,1,2,3,4,5))
'''


# #=========== Individual Model (more deterministic) with MI ===============
# # [0] Create one variable
# a = JointProbabilityMatrix(1, num_val,joint_probs="uniform")
# print(len(a))


# #[1][2] Append independent variables
# in_times = 2
# for jj in range(in_times):
#     b = JointProbabilityMatrix(1, num_val,joint_probs="uniform")
#     tf.append_independent_variables(a, b)
#     print(len(a))  
    
# # [3] Add symptoms with mutual information to diseases
# tf.append_variables_with_target_mi(a, 1, a.entropy([0]) * mi_factor, relevant_variables =[0])
# print(len(a))
    
# # [4] Append synergictic variables 0,1-> 4
# tf.append_variables_using_state_transitions_table(a,generate_XOR_transitions_table(a,dependent_vars=[0,1])) 

# # [5] Add symptoms with mutual information to diseases
# tf.append_variables_with_target_mi(a, 1, a.entropy([3,4]) * diagnosis_factor, relevant_variables =[3,4])
# print(len(a))

# # [6] 1->6
# tf.append_variables_with_target_mi(a, 1, a.entropy([1]) * mi_factor, relevant_variables =[1])
# print(len(a))

# # [7]
# tf.append_variables_with_target_mi(a, 1, a.entropy([6]) * diagnosis_factor, relevant_variables =[6])
# print(len(a))

# # [8] 
# tf.append_variables_using_state_transitions_table(a,generate_XOR_transitions_table(a,dependent_vars=[1,6])) 
    
# # [9]    
# tf.append_variables_with_target_mi(a, 1, a.entropy([2,8]) * diagnosis_factor, relevant_variables =[2,8])
# print(len(a))

# with open('simulations/toy_model_V'+str("XOR_with_MI_gates")+'.pickle', 'wb') as handle:
#     pickle.dump(a, handle, protocol=pickle.HIGHEST_PROTOCOL)

# #=========== Individual Model (more deterministic) with AND gate ===============
# # [0] Create one variable
# a = JointProbabilityMatrix(1, num_val,joint_probs="uniform")
# print(len(a))


# #[1][2] Append independent variables
# in_times = 2
# for jj in range(in_times):
#     b = JointProbabilityMatrix(1, num_val,joint_probs="uniform")
#     tf.append_independent_variables(a, b)
#     print(len(a))  
    
# # [3] Add symptoms with mutual information to diseases
# tf.append_variables_with_target_mi(a, 1, a.entropy([0]) * mi_factor, relevant_variables =[0])
# print(len(a))
    
# # [4] Append synergictic variables 0,1-> 4
# tf.append_variables_using_state_transitions_table(a,generate_XOR_transitions_table(a,dependent_vars=[0,1])) 

# # [5] Add symptoms with mutual information to diseases
# tf.append_variables_using_state_transitions_table(a,generate_AND_transitions_table(a,dependent_vars=[3,4])) 
# print(len(a))

# # [6] 1->6
# tf.append_variables_with_target_mi(a, 1, a.entropy([1]) * mi_factor, relevant_variables =[1])
# print(len(a))

# # [7]
# tf.append_variables_using_state_transitions_table(a,generate_AND_transitions_table(a,dependent_vars=[6])) 
# print(len(a))

# # [8] 
# tf.append_variables_using_state_transitions_table(a,generate_XOR_transitions_table(a,dependent_vars=[1,6])) 
    
# # [9]    
# tf.append_variables_using_state_transitions_table(a,generate_AND_transitions_table(a,dependent_vars=[2,8])) 
# print(len(a))
# with open('simulations/toy_model_V'+str("XOR_with_AND_gates")+'.pickle', 'wb') as handle:
#     pickle.dump(a, handle, protocol=pickle.HIGHEST_PROTOCOL)

# #=========== Individual Model (more deterministic) with OR gate ===============
# # [0] Create one variable
# a = JointProbabilityMatrix(1, num_val,joint_probs="uniform")
# print(len(a))


# #[1][2] Append independent variables
# in_times = 2
# for jj in range(in_times):
#     b = JointProbabilityMatrix(1, num_val,joint_probs="uniform")
#     tf.append_independent_variables(a, b)
#     print(len(a))  
    
# # [3] Add symptoms with mutual information to diseases
# tf.append_variables_with_target_mi(a, 1, a.entropy([0]) * mi_factor, relevant_variables =[0])
# print(len(a))
    
# # [4] Append synergictic variables 0,1-> 4
# tf.append_variables_using_state_transitions_table(a,generate_XOR_transitions_table(a,dependent_vars=[0,1])) 

# # [5] Add symptoms with mutual information to diseases
# tf.append_variables_using_state_transitions_table(a,generate_OR_transitions_table(a,dependent_vars=[3,4])) 
# print(len(a))

# # [6] 1->6
# tf.append_variables_with_target_mi(a, 1, a.entropy([1]) * mi_factor, relevant_variables =[1])
# print(len(a))

# # [7]
# tf.append_variables_using_state_transitions_table(a,generate_OR_transitions_table(a,dependent_vars=[6])) 
# print(len(a))

# # [8] 
# tf.append_variables_using_state_transitions_table(a,generate_XOR_transitions_table(a,dependent_vars=[1,6])) 
    
# # [9]    
# tf.append_variables_using_state_transitions_table(a,generate_OR_transitions_table(a,dependent_vars=[2,8])) 
# print(len(a))

# version_num = 0
# with open('simulations/toy_model_V'+str("XOR_with_OR_gates")+'.pickle', 'wb') as handle:
#     pickle.dump(a, handle, protocol=pickle.HIGHEST_PROTOCOL)