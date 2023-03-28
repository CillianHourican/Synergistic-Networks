# -*- coding: utf-8 -*-
"""
Created on Mon Mar 27 12:34:45 2023

@author: Cillian
"""
import numpy as np
import pickle
import pandas as pd
from JointProbabilityMatrix import JointProbabilityMatrix
import toy_functions as tf
from toy_symptom_model import generate_OR_transitions_table

num_var = 1
num_val = 4
num_sample = 100000
mi_factor = 0.9
diagnosis_factor = 0.9
source_prob_dist = "uniform"

a = JointProbabilityMatrix(1, num_val, joint_probs=source_prob_dist)
print(len(a))

b = JointProbabilityMatrix(1, num_val, joint_probs=source_prob_dist)
tf.append_independent_variables(a, b)

tf.append_synergistic_variables(a, 1, subject_variables=[0, 1])
print(len(a))

data = a.generate_samples(1000)
data = np.array(data)
pd.DataFrame(data).to_csv("synergistic_interaction_example.csv")

def Build_network(num_val=2, mi_factor=0.9, diagnosis_factor=0.9,
                  source_prob_dist="uniform", model_name="test"):

    # Todo: Generalise for a list of arrays
    # ToDo: Make sure valid istribution or np arrays are specified
    # ToDo: Use source_prob_dist input!! Currently removed, because not in use for this specific setup
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

    with open('Toy_model_disecete/toy_model'+str(model_name)+'.pickle', 'wb') as handle:
        pickle.dump(a, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return a

# Constructing the network may take some time
#Build_network()
