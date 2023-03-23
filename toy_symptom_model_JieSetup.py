# -*- coding: utf-8 -*-
"""
Created on Thu Mar 17 13:34:55 2022

@author: Cillian
"""
import numpy as np
import pickle
import multiprocess
import sys
import argparse
import time

import pandas as pd
from JointProbabilityMatrix import JointProbabilityMatrix
import toy_functions as tf
import numpy as np
from toy_symptom_model import generate_XOR_transitions_table, generate_AND_transitions_table, generate_OR_transitions_table

num_var = 1
num_val = 4
num_sample = 100000
mi_factor = 0.9
diagnosis_factor = 0.9


def Build_toy_model(num_val=2, mi_factor=0.9, diagnosis_factor=0.9, source_prob_dist="uniform", model_name="test", sema=None):

    # Todo: Generalise for a list of arrays
    # ToDo: Make sure valid istribution or np arrays are specified
    if type(source_prob_dist) == np.ndarray:
        assert source_prob_dist.sum() == 1, 'probability distribution should sum to one'

    start = time.time()

    full_attempt = 0
    while full_attempt < 5:
        full_attempt += 1

        try:
            half_attempt = 0
            while half_attempt < 10:
                half_attempt += 1

                try:
                    # [0] Create a diseases
                    a = JointProbabilityMatrix(
                        1, num_val, joint_probs=source_prob_dist[0])
                    print(len(a))

                    # [1][2] Append two independent variables (diseases)
                    in_times = 2
                    for jj in range(in_times):
                        b = JointProbabilityMatrix(
                            1, num_val, joint_probs=source_prob_dist[jj+1])
                        tf.append_independent_variables(a, b)
                        print(len(a))

                    # # [3] Add symptoms with mutual information to diseases
                    tf.append_variables_with_target_mi(
                        a, 1, a.entropy([0]) * mi_factor, relevant_variables=[0])
                    print(len(a))

                    # [4] Append synergictic variables 0,1-> 4
                    attempt = 0
                    while len(a) < 5 and attempt < 30:
                        attempt += 1
                        try:
                            tf.append_synergistic_variables(
                                a, 1, subject_variables=[0, 1])
                            print(len(a))
                        except:
                            print(
                                "Failed to add synergistic variable [4]. Trying again..")

                except:
                    print(
                        "Restarting construction attempt.. kept failing to add synergistic variable [4]...")
            # try:
            #     tf.append_synergistic_variables(a, 1, subject_variables=[0,1])
            # except UserWarning as e:
            #     assert 'minimize() failed'
            # print(len(a))

            # [5] Add symptoms with mutual information to diseases
            tf.append_variables_with_target_mi(a, 1, a.entropy(
                [3, 4]) * diagnosis_factor, relevant_variables=[3, 4])
            print(len(a))

            # [6] 1->6
            tf.append_variables_with_target_mi(
                a, 1, a.entropy([1]) * mi_factor, relevant_variables=[1])
            print(len(a))

            # [7]
            tf.append_variables_with_target_mi(a, 1, a.entropy(
                [6]) * diagnosis_factor, relevant_variables=[6])
            print(len(a))

            #num_vars = 8
            attempt = 0
            while len(a) < 9 and attempt < 30:
                attempt += 1
                # Append two synergictic variables 1,6 -> 8
                try:
                    tf.append_synergistic_variables(
                        a, 1, subject_variables=[1, 6])
                    print(len(a))
                except:
                    print("Failed to add synergistic variable. Trying again..")

            # [9]
            tf.append_variables_with_target_mi(a, 1, a.entropy(
                [2, 8]) * diagnosis_factor, relevant_variables=[2, 8])
            print(len(a))

        except:
            print("Failed. Starting again...")
            if full_attempt == 5 and sema:
                sema.release()

    end = time.time()
    a.build_time = end-start

    with open('simulations/toy_model_V_'+str(model_name)+'.pickle', 'wb') as handle:
        pickle.dump(a, handle, protocol=pickle.HIGHEST_PROTOCOL)

    if sema:
        sema.release()

    return a


def Build_smaller_toy_model(num_val=2, mi_factor=0.9, diagnosis_factor=0.9, source_prob_dist="uniform", model_name="test", sema=None):

    # Todo: Generalise for a list of arrays
    # ToDo: Make sure valid istribution or np arrays are specified
    if type(source_prob_dist) == np.ndarray:
        assert source_prob_dist.sum() == 1, 'probability distribution should sum to one'

    start = time.time()
    a = None
    full_attempt = 0
    while full_attempt < 5:
        full_attempt += 1

        try:
            half_attempt = 0
            while half_attempt < 10:
                half_attempt += 1

                try:
                    # [0] Create a diseases
                    a = JointProbabilityMatrix(
                        1, num_val, joint_probs=source_prob_dist)
                    print(len(a))

                    # [1][2] Append two independent variables (diseases)
                    in_times = 2
                    for jj in range(in_times):
                        b = JointProbabilityMatrix(
                            1, num_val, joint_probs=source_prob_dist)
                        tf.append_independent_variables(a, b)
                        print(len(a))

                    # # # [3] Add symptoms with mutual information to diseases
                    # tf.append_variables_with_target_mi(a, 1, a.entropy([0]) * mi_factor, relevant_variables =[0])
                    # print(len(a))

                    # [3] Append synergictic variables 0,1-> 3
                    attempt = 0
                    while len(a) < 5 and attempt < 30:
                        attempt += 1
                        try:
                            tf.append_synergistic_variables(
                                a, 1, subject_variables=[0, 1])
                            print(len(a))
                        except:
                            print(
                                "Failed to add synergistic variable [4]. Trying again..")

                except:
                    print(
                        "Restarting construction attempt.. kept failing to add synergistic variable [4]...")
            # try:
            #     tf.append_synergistic_variables(a, 1, subject_variables=[0,1])
            # except UserWarning as e:
            #     assert 'minimize() failed'
            # print(len(a))

            # [4] Add symptoms with mutual information to diseases
            tf.append_variables_with_target_mi(a, 1, a.entropy(
                [3, 0]) * diagnosis_factor, relevant_variables=[3, 0])
            print(len(a))

            # [5] 1->5
            tf.append_variables_with_target_mi(
                a, 1, a.entropy([1]) * mi_factor, relevant_variables=[1])
            print(len(a))

            # [6]
            tf.append_variables_with_target_mi(a, 1, a.entropy(
                [5]) * diagnosis_factor, relevant_variables=[5])
            print(len(a))

            #num_vars = 7
            attempt = 0
            while len(a) < 8 and attempt < 30:
                attempt += 1

                try:
                    tf.append_synergistic_variables(
                        a, 1, subject_variables=[1, 5])
                    print(len(a))
                except:
                    print("Failed to add synergistic variable. Trying again..")

            # [8]
            tf.append_variables_with_target_mi(a, 1, a.entropy(
                [2, 7]) * diagnosis_factor, relevant_variables=[2, 7])
            print(len(a))

        except:
            print("Failed. Starting again...")
            if full_attempt == 5 and sema:
                sema.release()

    end = time.time()
    #a.build_time = end-start

    with open('Toy_model_disecete/V2_toy_model_V_'+str(model_name)+'.pickle', 'wb') as handle:
        pickle.dump(a, handle, protocol=pickle.HIGHEST_PROTOCOL)

    if sema:
        sema.release()

    return a


def Build_correlation_toy_model(num_val=2, factor=0.9, diagnosis_factor=0.8, source_prob_dist="uniform", model_name="test", sema=None):

    # Todo: Generalise for a list of arrays
    # ToDo: Make sure valid istribution or np arrays are specified
    if type(source_prob_dist) == np.ndarray:
        assert source_prob_dist.sum() == 1, 'probability distribution should sum to one'

    start = time.time()

    full_attempt = 0
    while full_attempt < 5:
        full_attempt += 1

        try:
            half_attempt = 0
            while half_attempt < 10:
                half_attempt += 1

                try:
                    # [0] Create a diseases
                    a = JointProbabilityMatrix(
                        1, num_val, joint_probs=source_prob_dist[0])
                    print(len(a))

                    # [1][2] Append two independent variables (diseases)
                    in_times = 2
                    for jj in range(in_times):
                        b = JointProbabilityMatrix(
                            1, num_val, joint_probs=source_prob_dist[jj+1])
                        tf.append_independent_variables(a, b)
                        print(len(a))

                    # # # [3] Add symptoms with mutual information to diseases
                    # tf.append_variables_with_target_mi(a, 1, a.entropy([0]) * mi_factor, relevant_variables =[0])
                    # print(len(a))

                    # [3] Append synergictic variables 0,1-> 3
                    attempt = 0
                    while len(a) < 5 and attempt < 30:
                        attempt += 1
                        try:
                            tf.append_synergistic_variables(
                                a, 1, subject_variables=[0, 1])
                            print(len(a))
                        except:
                            print(
                                "Failed to add synergistic variable [4]. Trying again..")

                except:
                    print(
                        "Restarting construction attempt.. kept failing to add synergistic variable [4]...")
            # try:
            #     tf.append_synergistic_variables(a, 1, subject_variables=[0,1])
            # except UserWarning as e:
            #     assert 'minimize() failed'
            # print(len(a))

            # [4] Add symptoms with mutual information to diseases
            tf.append_variables_with_target_corr(
                a, 1, diagnosis_factor, relevant_variables=[3, 0])
            print(len(a))

            # [5] 1->5
            tf.append_variables_with_target_corr(
                a, 1, -diagnosis_factor, relevant_variables=[1])
            print(len(a))

            # [6]
            tf.append_variables_with_target_corr(
                a, 1, diagnosis_factor, relevant_variables=[5])
            print(len(a))

            #num_vars = 7
            attempt = 0
            while len(a) < 8 and attempt < 30:
                attempt += 1

                try:
                    tf.append_synergistic_variables(
                        a, 1, subject_variables=[1, 5])
                    print(len(a))
                except:
                    print("Failed to add synergistic variable. Trying again..")

            # [8]
            tf.append_variables_with_target_corr(
                a, 1, diagnosis_factor, relevant_variables=[2, 7])
            print(len(a))

        except:
            print("Failed. Starting again...")
            if full_attempt == 5 and sema:
                sema.release()

    end = time.time()
    a.build_time = end-start

    with open('simulations/corrs_V1_toy_model_V_'+str(model_name)+'.pickle', 'wb') as handle:
        pickle.dump(a, handle, protocol=pickle.HIGHEST_PROTOCOL)

    if sema:
        sema.release()

    return a


def Build_multi_correlation_toy_model(num_val=2, factor=0.9, diagnosis_factor=0.8, source_prob_dist="uniform", model_name="test", sema=None):

    # Todo: Generalise for a list of arrays
    # ToDo: Make sure valid istribution or np arrays are specified
    if type(source_prob_dist) == np.ndarray:
        assert source_prob_dist.sum() == 1, 'probability distribution should sum to one'

    start = time.time()

    full_attempt = 0
    while full_attempt < 5:
        full_attempt += 1

        try:
            half_attempt = 0
            while half_attempt < 10:
                half_attempt += 1

                try:
                    # [0] Create a diseases
                    a = JointProbabilityMatrix(
                        1, num_val, joint_probs=source_prob_dist[0])
                    print(len(a))

                    # [1][2] Append two independent variables (diseases)
                    in_times = 2
                    for jj in range(in_times):
                        b = JointProbabilityMatrix(
                            1, num_val, joint_probs=source_prob_dist[jj+1])
                        tf.append_independent_variables(a, b)
                        print(len(a))

                    # # # [3] Add symptoms with mutual information to diseases
                    # tf.append_variables_with_target_mi(a, 1, a.entropy([0]) * mi_factor, relevant_variables =[0])
                    # print(len(a))

                    # [3] Append synergictic variables 0,1-> 3
                    attempt = 0
                    while len(a) < 5 and attempt < 30:
                        attempt += 1
                        try:
                            tf.append_synergistic_variables(
                                a, 1, subject_variables=[0, 1])
                            print(len(a))
                        except:
                            print(
                                "Failed to add synergistic variable [4]. Trying again..")

                except:
                    print(
                        "Restarting construction attempt.. kept failing to add synergistic variable [4]...")
            # try:
            #     tf.append_synergistic_variables(a, 1, subject_variables=[0,1])
            # except UserWarning as e:
            #     assert 'minimize() failed'
            # print(len(a))

            # [4] Add symptoms with mutual information to diseases
            tf.append_variables_with_target_corr(
                a, 1, [0.8, 0.4], relevant_variables=[3, 0])
            print(len(a))

            # [5] 1->5
            tf.append_variables_with_target_corr(
                a, 1, -0.8, relevant_variables=[1])
            print(len(a))

            # [6]
            tf.append_variables_with_target_corr(
                a, 1, 0.6, relevant_variables=[5])
            print(len(a))

            #num_vars = 7
            attempt = 0
            while len(a) < 8 and attempt < 30:
                attempt += 1

                try:
                    tf.append_synergistic_variables(
                        a, 1, subject_variables=[1, 5])
                    print(len(a))
                except:
                    print("Failed to add synergistic variable. Trying again..")

            # [8]
            tf.append_variables_with_target_corr(
                a, 1, [0.2, 0.8], relevant_variables=[2, 7])
            print(len(a))

        except:
            print("Failed. Starting again...")
            if full_attempt == 5 and sema:
                sema.release()

    end = time.time()
    a.build_time = end-start

    with open('simulations/multicorrs_V1_toy_model_V_'+str(model_name)+'.pickle', 'wb') as handle:
        pickle.dump(a, handle, protocol=pickle.HIGHEST_PROTOCOL)

    if sema:
        sema.release()

    return a


# def save_model(model_name):
#     with open('toy_model_V_'+str(model_name)+'.pickle', 'wb') as handle:
#         pickle.dump(a, handle, protocol=pickle.HIGHEST_PROTOCOL)

# # Save network

# model_name = "n0_4060"
# with open('toy_model_V'+str(model_name)+'.pickle', 'wb') as handle:
#     pickle.dump(a, handle, protocol=pickle.HIGHEST_PROTOCOL)

# original1 = Build_toy_model(num_val = 2, mi_factor = 0.9, diagnosis_factor = 0.9, source_prob_dist=["uniform","uniform","uniform"])
# original1_v2 = Build_toy_model(source_prob_dist=[np.array([0.5,0.5]),np.array([0.5,0.5]),np.array([0.5,0.5])])

# model_name = "n0_4060"
# intervention1 = Build_toy_model(source_prob_dist=[np.array([0.4,0.6]),"uniform","uniform"])

# model_name = "n0_3070"
# intervention2 = Build_toy_model(source_prob_dist=[np.array([0.3,0.7]),"uniform","uniform"]) # Fails

# model_name = "n1_4060"
# intervention1 = Build_toy_model(source_prob_dist=["uniform",np.array([0.4,0.6]),"uniform"])

# model_name = "n123_4060"
# intervention2 = Build_toy_model(source_prob_dist=[np.array([0.4,0.6]),np.array([0.4,0.6]),np.array([0.4,0.6])])

# Note: This fails with some combinations!

#individual_network = Build_toy_model(source_prob_dist=[np.array([0,1]),np.array([0,1]),np.array([0,1])])

# Build_toy_model(source_prob_dist=[np.array([0.01,0.99]),np.array([0.01,0.99]),np.array([0.01,0.99])])

# tf.append_variables_using_state_transitions_table(a,[[0,0],[1,1]])

#############################################################################################
# # [0] Create three diseases -> should they be 3 "independent" variables?
# aa = JointProbabilityMatrix(1, num_val,joint_probs=np.array([0.3,0.5]))
# print(len(aa))

# # # Add symptoms with correlation to symptom
# # tf.append_variables_with_target_corr(a, 1, 0.5, relevant_variables =[0])
# # print(len(a))


# #[1][2] Append independent variables
# in_times = 2
# for jj in range(in_times):
#     bb = JointProbabilityMatrix(1, num_val,joint_probs=np.array([0,1]))
#     tf.append_independent_variables(aa, bb)
#     print(len(aa))


# import pandas as pd
# from JointProbabilityMatrix import JointProbabilityMatrix
# import toy_functions as tf
# import numpy as np

# num_var = 1
# num_val = 2
# num_sample = 100000
# mi_factor = 0.9
# diagnosis_factor = 0.9

def Build_toy_model_OR(num_val=2, mi_factor=0.9, diagnosis_factor=0.9, source_prob_dist="uniform", model_name="test", sema=None):

    # Todo: Generalise for a list of arrays
    # ToDo: Make sure valid istribution or np arrays are specified
    if type(source_prob_dist) == np.ndarray:
        assert source_prob_dist.sum() == 1, 'probability distribution should sum to one'

    start = time.time()

    full_attempt = 0
    while full_attempt < 5:
        full_attempt += 1

        try:
            half_attempt = 0
            while half_attempt < 10:
                half_attempt += 1

                try:
                    # [0] Create a node
                    a = JointProbabilityMatrix(
                        1, num_val, joint_probs=source_prob_dist[0])
                    # print(len(a))

                    # [1][2] Append two independent variables (diseases)
                    in_times = 2
                    for jj in range(in_times):
                        b = JointProbabilityMatrix(
                            1, num_val, joint_probs=source_prob_dist[jj+1])
                        tf.append_independent_variables(a, b)
                        # print(len(a))

                    # # # [3] Add symptoms with mutual information to diseases
                    # tf.append_variables_with_target_mi(a, 1, a.entropy([0]) * mi_factor, relevant_variables =[0])
                    # print(len(a))

                    # [3] Append synergictic variables 0,1-> 3
                    attempt = 0
                    while len(a) < 4 and attempt < 30:
                        attempt += 1
                        try:
                            tf.append_synergistic_variables(
                                a, 1, subject_variables=[0, 1])
                            # print(len(a))
                        except:
                            print(
                                "Failed to add synergistic variable [4]. Trying again..")

                except:
                    print(
                        "Restarting construction attempt.. kept failing to add synergistic variable [4]...")
            # try:
            #     tf.append_synergistic_variables(a, 1, subject_variables=[0,1])
            # except UserWarning as e:
            #     assert 'minimize() failed'
            # print(len(a))

            # [4] Add symptoms with mutual information to diseases
            #tf.append_variables_with_target_mi(a, 1, a.entropy([3,0]) * diagnosis_factor, relevant_variables =[3,0])
            tf.append_variables_using_state_transitions_table(
                a, generate_OR_transitions_table(a, dependent_vars=[3, 0]))
            # print(len(a))

            # [5] 1->5
            tf.append_variables_with_target_mi(
                a, 1, a.entropy([1]) * mi_factor, relevant_variables=[1])
            # print(len(a))

            # [6]
            tf.append_variables_with_target_mi(a, 1, a.entropy(
                [5]) * diagnosis_factor, relevant_variables=[5])
            # print(len(a))

            #num_vars = 7
            attempt = 0
            while len(a) < 8 and attempt < 30:
                attempt += 1

                try:
                    #tf.append_synergistic_variables(a, 1, subject_variables=[1,5])
                    tf.append_variables_using_state_transitions_table(
                        a, generate_OR_transitions_table(a, dependent_vars=[1, 5]))
                    # print(len(a))
                except:
                    print("Failed to add synergistic variable. Trying again..", len(a))

            # [8]
            #tf.append_variables_with_target_mi(a, 1, a.entropy([2,7]) * diagnosis_factor, relevant_variables =[2,7])
            tf.append_variables_using_state_transitions_table(
                a, generate_OR_transitions_table(a, dependent_vars=[2, 7]))
            # print(len(a))

        except:
            print("Failed. Starting again...")
            if full_attempt == 5 and sema:
                sema.release()

    end = time.time()
    a.build_time = end-start

    with open('simulations/V2_toy_model_OR_7OR_'+str(model_name)+'.pickle', 'wb') as handle:
        pickle.dump(a, handle, protocol=pickle.HIGHEST_PROTOCOL)

    if sema:
        sema.release()

    return a


def Build_toy_model_XOR(num_val=2, mi_factor=0.9, diagnosis_factor=0.9, source_prob_dist="uniform", model_name="test", sema=None):

    # Todo: Generalise for a list of arrays
    # ToDo: Make sure valid istribution or np arrays are specified
    if type(source_prob_dist) == np.ndarray:
        assert source_prob_dist.sum() == 1, 'probability distribution should sum to one'

    start = time.time()

    full_attempt = 0
    while full_attempt < 5:
        full_attempt += 1

        try:
            half_attempt = 0
            while half_attempt < 10:
                half_attempt += 1

                try:
                    # [0] Create a diseases
                    a = JointProbabilityMatrix(
                        1, num_val, joint_probs=source_prob_dist[0])
                    print(len(a))

                    # [1][2] Append two independent variables (diseases)
                    in_times = 2
                    for jj in range(in_times):
                        b = JointProbabilityMatrix(
                            1, num_val, joint_probs=source_prob_dist[jj+1])
                        tf.append_independent_variables(a, b)
                        print(len(a))

                    # # # [3] Add symptoms with mutual information to diseases
                    # tf.append_variables_with_target_mi(a, 1, a.entropy([0]) * mi_factor, relevant_variables =[0])
                    # print(len(a))

                    # [3] Append synergictic variables 0,1-> 3
                    attempt = 0
                    while len(a) < 4 and attempt < 30:
                        attempt += 1
                        try:
                            tf.append_synergistic_variables(
                                a, 1, subject_variables=[0, 1])
                            print(len(a))
                        except:
                            print(
                                "Failed to add synergistic variable [4]. Trying again..")

                except:
                    print(
                        "Restarting construction attempt.. kept failing to add synergistic variable [4]...")
            # try:
            #     tf.append_synergistic_variables(a, 1, subject_variables=[0,1])
            # except UserWarning as e:
            #     assert 'minimize() failed'
            # print(len(a))

            # [4] Add symptoms with mutual information to diseases
            #tf.append_variables_with_target_mi(a, 1, a.entropy([3,0]) * diagnosis_factor, relevant_variables =[3,0])
            tf.append_variables_using_state_transitions_table(
                a, generate_XOR_transitions_table(a, dependent_vars=[3, 0]))
            print(len(a))

            # [5] 1->5
            tf.append_variables_with_target_mi(
                a, 1, a.entropy([1]) * mi_factor, relevant_variables=[1])
            print(len(a))

            # [6]
            tf.append_variables_with_target_mi(a, 1, a.entropy(
                [5]) * diagnosis_factor, relevant_variables=[5])
            print(len(a))

            #num_vars = 7
            attempt = 0
            while len(a) < 8 and attempt < 30:
                attempt += 1

                try:
                    tf.append_synergistic_variables(
                        a, 1, subject_variables=[1, 5])
                    print(len(a))
                except:
                    print("Failed to add synergistic variable. Trying again..", len(a))

            # [8]
            #tf.append_variables_with_target_mi(a, 1, a.entropy([2,7]) * diagnosis_factor, relevant_variables =[2,7])
            tf.append_variables_using_state_transitions_table(
                a, generate_XOR_transitions_table(a, dependent_vars=[2, 7]))
            print(len(a))

        except:
            print("Failed. Starting again...")
            if full_attempt == 5 and sema:
                sema.release()

    end = time.time()
    a.build_time = end-start

    with open('simulations/V2_toy_model_XOR_'+str(model_name)+'.pickle', 'wb') as handle:
        pickle.dump(a, handle, protocol=pickle.HIGHEST_PROTOCOL)

    if sema:
        sema.release()

    return a


def Build_AND_toy_model(num_val=2, mi_factor=0.9, diagnosis_factor=0.9, source_prob_dist="uniform", model_name="test", sema=None):

    # Todo: Generalise for a list of arrays
    # ToDo: Make sure valid istribution or np arrays are specified
    if type(source_prob_dist) == np.ndarray:
        assert source_prob_dist.sum() == 1, 'probability distribution should sum to one'

    start = time.time()

    full_attempt = 0
    while full_attempt < 5:
        full_attempt += 1

        try:
            half_attempt = 0
            while half_attempt < 10:
                half_attempt += 1

                try:
                    # [0] Create a diseases
                    a = JointProbabilityMatrix(
                        1, num_val, joint_probs=source_prob_dist[0])
                    print(len(a))

                    # [1][2] Append two independent variables (diseases)
                    in_times = 2
                    for jj in range(in_times):
                        b = JointProbabilityMatrix(
                            1, num_val, joint_probs=source_prob_dist[jj+1])
                        tf.append_independent_variables(a, b)
                        print(len(a))

                    # # # [3] Add symptoms with mutual information to diseases
                    # tf.append_variables_with_target_mi(a, 1, a.entropy([0]) * mi_factor, relevant_variables =[0])
                    # print(len(a))

                    # [3] Append synergictic variables 0,1-> 3
                    attempt = 0
                    while len(a) < 5 and attempt < 30:
                        attempt += 1
                        try:
                            tf.append_synergistic_variables(
                                a, 1, subject_variables=[0, 1])
                            print(len(a))
                        except:
                            print(
                                "Failed to add synergistic variable [4]. Trying again..")

                except:
                    print(
                        "Restarting construction attempt.. kept failing to add synergistic variable [4]...")
            # try:
            #     tf.append_synergistic_variables(a, 1, subject_variables=[0,1])
            # except UserWarning as e:
            #     assert 'minimize() failed'
            # print(len(a))

            # [4] Add symptoms with mutual information to diseases
            #tf.append_variables_with_target_mi(a, 1, a.entropy([3,0]) * diagnosis_factor, relevant_variables =[3,0])
            tf.append_variables_using_state_transitions_table(
                a, generate_AND_transitions_table(a, dependent_vars=[3, 0]))
            print(len(a))

            # [5] 1->5
            tf.append_variables_with_target_mi(
                a, 1, a.entropy([1]) * mi_factor, relevant_variables=[1])
            print(len(a))

            # [6]
            tf.append_variables_with_target_mi(a, 1, a.entropy(
                [5]) * diagnosis_factor, relevant_variables=[5])
            print(len(a))

            #num_vars = 7
            attempt = 0
            while len(a) < 8 and attempt < 30:
                attempt += 1

                try:
                    tf.append_synergistic_variables(
                        a, 1, subject_variables=[1, 5])
                    print(len(a))
                except:
                    print("Failed to add synergistic variable. Trying again..")

            # [8]
            #tf.append_variables_with_target_mi(a, 1, a.entropy([2,7]) * diagnosis_factor, relevant_variables =[2,7])
            tf.append_variables_using_state_transitions_table(
                a, generate_AND_transitions_table(a, dependent_vars=[2, 7]))
            print(len(a))

        except:
            print("Failed. Starting again...")
            if full_attempt == 5 and sema:
                sema.release()

    end = time.time()
    a.build_time = end-start

    with open('simulations/V2_toy_model_AND_'+str(model_name)+'.pickle', 'wb') as handle:
        pickle.dump(a, handle, protocol=pickle.HIGHEST_PROTOCOL)

    if sema:
        sema.release()

    return a


def Build_re_ordered_toy_model_OR(num_val=2, mi_factor=0.9, diagnosis_factor=0.9, source_prob_dist="uniform", model_name="test", sema=None):

    # Todo: Generalise for a list of arrays
    # ToDo: Make sure valid istribution or np arrays are specified
    # ToDo: Use source_prob_dist input!! Currently removed, because not in use for this specific setup
    if type(source_prob_dist) == np.ndarray:
        assert source_prob_dist.sum() == 1, 'probability distribution should sum to one'

    start = time.time()

    first_half_attempt = 0

    a = [None]  # dummy variable
    while (len(a) < 3) and (first_half_attempt < 10):
        try:
            # [0] B
            a = JointProbabilityMatrix(1, num_val)

            # [1] C
            tf.append_variables_with_target_mi(
                a, 1, a.entropy([0]) * mi_factor, relevant_variables=[0])

            # [2] F
            # tf.append_variables_using_state_transitions_table(a,generate_OR_transitions_table(a,dependent_vars=[0,1]))
            attempt = 0
            while (len(a) < 3) and (attempt < 10):
                try:
                    tf.append_synergistic_variables(
                        a, 1, subject_variables=[0, 1])
                except:
                    print("Failed attempt "+str(attempt) +
                          " trying to append synergistic variable F..")
                    attempt += 1
        except:
            print("Failed appening firsy synergistic var. Starting again. This is attempt " +
                  str(first_half_attempt)+"...")
            first_half_attempt += 1

    # [3] I
    tf.append_variables_with_target_mi(
        a, 1, a.entropy([1]) * mi_factor, relevant_variables=[1])

    # [4]
    b = JointProbabilityMatrix(1, num_val)
    tf.append_independent_variables(a, b)

    attempt = 0
    while (len(a) < 6) and (attempt < 40):
        try:
            tf.append_synergistic_variables(a, 1, subject_variables=[0, 4])
        except:
            print("Failed attempt "+str(attempt) +
                  " trying to append synergistic variable D..")
            attempt += 1
    # tf.append_variables_using_state_transitions_table(a,generate_OR_transitions_table(a,dependent_vars=[0,4]))
    tf.append_variables_using_state_transitions_table(
        a, generate_OR_transitions_table(a, dependent_vars=[4, 5]))

    # [7]
    b = JointProbabilityMatrix(1, num_val)
    tf.append_independent_variables(a, b)

    # [8]
    tf.append_variables_using_state_transitions_table(
        a, generate_OR_transitions_table(a, dependent_vars=[7, 2]))

    end = time.time()
    a.build_time = end-start

    with open('Toy_model_disecete/V2_toy_model_OR_rearranged_'+str(model_name)+'.pickle', 'wb') as handle:
        pickle.dump(a, handle, protocol=pickle.HIGHEST_PROTOCOL)

    if sema:
        sema.release()

    return a


def Build_re_ordered_toy_model_OR_uniform(num_val=2, mi_factor=0.9, diagnosis_factor=0.9, source_prob_dist="uniform", model_name="test", sema=None):

    # Todo: Generalise for a list of arrays
    # ToDo: Make sure valid istribution or np arrays are specified
    # ToDo: Use source_prob_dist input!! Currently removed, because not in use for this specific setup
    if type(source_prob_dist) == np.ndarray:
        assert source_prob_dist.sum() == 1, 'probability distribution should sum to one'

    start = time.time()

    first_half_attempt = 0

    a = [None]  # dummy variable
    while (len(a) < 3) and (first_half_attempt < 10):
        try:
            # [0] B
            a = JointProbabilityMatrix(1, num_val, joint_probs="uniform")

            # [1] C
            tf.append_variables_with_target_mi(
                a, 1, a.entropy([0]) * mi_factor, relevant_variables=[0])

            # [2] F
            # tf.append_variables_using_state_transitions_table(a,generate_OR_transitions_table(a,dependent_vars=[0,1]))
            attempt = 0
            while (len(a) < 3) and (attempt < 10):
                try:
                    tf.append_synergistic_variables(
                        a, 1, subject_variables=[0, 1])
                except:
                    print("Failed attempt "+str(attempt) +
                          " trying to append synergistic variable F..")
                    attempt += 1
        except:
            print("Failed appening firsy synergistic var. Starting again. This is attempt " +
                  str(first_half_attempt)+"...")
            first_half_attempt += 1

    # [3] I
    tf.append_variables_with_target_mi(
        a, 1, a.entropy([1]) * mi_factor, relevant_variables=[1])

    # [4]
    b = JointProbabilityMatrix(1, num_val, joint_probs="uniform")
    tf.append_independent_variables(a, b)

    attempt = 0
    while (len(a) < 6) and (attempt < 40):
        try:
            tf.append_synergistic_variables(a, 1, subject_variables=[0, 4])
        except:
            print("Failed attempt "+str(attempt) +
                  " trying to append synergistic variable D..")
            attempt += 1
    # tf.append_variables_using_state_transitions_table(a,generate_OR_transitions_table(a,dependent_vars=[0,4]))
    tf.append_variables_using_state_transitions_table(
        a, generate_OR_transitions_table(a, dependent_vars=[4, 5]))

    # [7]
    b = JointProbabilityMatrix(1, num_val, joint_probs="uniform")
    tf.append_independent_variables(a, b)

    # [8]
    tf.append_variables_using_state_transitions_table(
        a, generate_OR_transitions_table(a, dependent_vars=[7, 2]))

    end = time.time()
    a.build_time = end-start

    with open('simulations/V2_toy_model_OR_rearranged_uniform_'+str(model_name)+'.pickle', 'wb') as handle:
        pickle.dump(a, handle, protocol=pickle.HIGHEST_PROTOCOL)

    if sema:
        sema.release()

    return a


def Build_re_ordered_toy_model_OR_XORsynergy(num_val=2, mi_factor=0.9, diagnosis_factor=0.9, source_prob_dist="uniform", model_name="test", sema=None):

    # Todo: Generalise for a list of arrays
    # ToDo: Make sure valid istribution or np arrays are specified
    # ToDo: Use source_prob_dist input!! Currently removed, because not in use for this specific setup
    if type(source_prob_dist) == np.ndarray:
        assert source_prob_dist.sum() == 1, 'probability distribution should sum to one'

    start = time.time()

    first_half_attempt = 0

    a = [None]  # dummy variable
    while (len(a) < 3) and (first_half_attempt < 10):
        try:
            # [0] B
            a = JointProbabilityMatrix(1, num_val, joint_probs="uniform")

            # [1] C
            tf.append_variables_with_target_mi(
                a, 1, a.entropy([0]) * mi_factor, relevant_variables=[0])

            # [2] F
            tf.append_variables_using_state_transitions_table(
                a, generate_XOR_transitions_table(a, dependent_vars=[0, 1]))

        except:
            print("Failed appening firsy synergistic var. Starting again. This is attempt " +
                  str(first_half_attempt)+"...")
            first_half_attempt += 1

    # [3] I
    tf.append_variables_with_target_mi(
        a, 1, a.entropy([1]) * mi_factor, relevant_variables=[1])

    # [4] A
    b = JointProbabilityMatrix(1, num_val, joint_probs="uniform")
    tf.append_independent_variables(a, b)

    # [5] D
    tf.append_variables_using_state_transitions_table(
        a, generate_XOR_transitions_table(a, dependent_vars=[0, 4]))

    # [6] G
    tf.append_variables_using_state_transitions_table(
        a, generate_OR_transitions_table(a, dependent_vars=[4, 5]))

    # [7] E
    b = JointProbabilityMatrix(1, num_val, joint_probs="uniform")
    tf.append_independent_variables(a, b)

    # [8]
    tf.append_variables_using_state_transitions_table(
        a, generate_OR_transitions_table(a, dependent_vars=[7, 2]))

    end = time.time()
    a.build_time = end-start

    with open('simulations/V2_toy_model_OR_rearranged_XORsynergy_uniform'+str(model_name)+'.pickle', 'wb') as handle:
        pickle.dump(a, handle, protocol=pickle.HIGHEST_PROTOCOL)

    if sema:
        sema.release()

    return a


def Build_re_ordered_toy_model_OR_XORsynergy2(num_val=2, mi_factor=0.9, diagnosis_factor=0.9, source_prob_dist="uniform", model_name="test", sema=None):

    # Todo: Generalise for a list of arrays
    # ToDo: Make sure valid istribution or np arrays are specified
    # ToDo: Use source_prob_dist input!! Currently removed, because not in use for this specific setup
    if type(source_prob_dist) == np.ndarray:
        assert source_prob_dist.sum() == 1, 'probability distribution should sum to one'

    start = time.time()

    first_half_attempt = 0

    a = [None]  # dummy variable
    while (len(a) < 3) and (first_half_attempt < 10):
        try:
            # [0] B
            a = JointProbabilityMatrix(1, num_val)

            # [1] C
            tf.append_variables_with_target_mi(
                a, 1, a.entropy([0]) * mi_factor, relevant_variables=[0])

            # [2] F
            tf.append_variables_using_state_transitions_table(
                a, generate_XOR_transitions_table(a, dependent_vars=[0, 1]))

        except:
            print("Failed appening firsy synergistic var. Starting again. This is attempt " +
                  str(first_half_attempt)+"...")
            first_half_attempt += 1

    # [3] I
    tf.append_variables_with_target_mi(
        a, 1, a.entropy([1]) * mi_factor, relevant_variables=[1])

    # [4] A
    b = JointProbabilityMatrix(1, num_val)
    tf.append_independent_variables(a, b)

    # [5] D
    tf.append_variables_using_state_transitions_table(
        a, generate_XOR_transitions_table(a, dependent_vars=[0, 4]))

    # [6] G
    tf.append_variables_using_state_transitions_table(
        a, generate_OR_transitions_table(a, dependent_vars=[4, 5]))

    # [7] E
    b = JointProbabilityMatrix(1, num_val)
    tf.append_independent_variables(a, b)

    # [8]
    tf.append_variables_using_state_transitions_table(
        a, generate_OR_transitions_table(a, dependent_vars=[7, 2]))

    end = time.time()
    a.build_time = end-start

    with open('simulations/V2_toy_model_OR_rearranged_XORsynergy_'+str(model_name)+'.pickle', 'wb') as handle:
        pickle.dump(a, handle, protocol=pickle.HIGHEST_PROTOCOL)

    if sema:
        sema.release()

    return a
