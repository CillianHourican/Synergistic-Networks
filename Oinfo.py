# -*- coding: utf-8 -*-
"""
Created on Fri Aug 19 12:36:27 2022

@author: Cillian
"""

import pandas as pd
from npeet import entropy_estimators as ee
import itertools
import tqdm


def compute_single(data):
    '''
    Computes the Oinfo score for discrete variables. Data is a pandas dataframe
    [num_observations, num_variables]

    Parameters
    ----------
    data : pandas dataframs
        Shape [num_observations, num_variables]

    Returns
    -------
    numpy.float64

    '''
    o_info = (len(data.columns) - 2)*ee.entropyd(data)
    for j, _ in enumerate(data.columns):

        o_info += ee.entropyd(data.loc[:, data.columns == _])
        o_info -= ee.entropyd(data.loc[:, data.columns != _])

    return(o_info)


def compute_all(data, num_vars=3):
    """
    Computes Oinfo scores for all groups of a certain size. Note that this
    computes everything sequentially. It may be slow for large datasets.

    Parameters
    ----------
    data : pandas dataframe
        Shape [num_observations, num_variables].
    num_vars : integer, optional
        Size of groups to compute Oinfo scores. The default is 3.

    Returns
    -------
    Pandas dataframe (default) or list.

    """

    triplets = list(itertools.combinations(
        range(len(data.columns)), num_vars))[:100_000]
    synergistic_triplets = []

    for triplet in tqdm(triplets):
        o_info = compute_single(data[data.columns[[triplet]]])
        synergistic_triplets.append(o_info)

    d = {'Triplet': triplets, 'Oinfo': synergistic_triplets}
    df = pd.DataFrame(d)

    df.to_csv("data/Oinfo_"+str(num_vars)+".csv")


df = pd.read_csv("data/synthetic_model_data.csv")

# Order of associations to compute
N = 3

compute_all(df, N)
