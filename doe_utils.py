"""
utils for DOE 
"""

__copyright__ = " "
__status__ = "Dev"
__doc__ = """ unilities """
__author__  = "Swagatam Mukhopadhyay"

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from Bio.Data import IUPACData
from collections import defaultdict
from scipy.spatial.distance import hamming
import itertools

def hamming_dist(seq1, seq2): 
    """
    compute hamming distance (number of mutations) for same length sequences 
    """
    return len(seq1)*hamming(list(seq1), list(seq2))


def get_n_choose_m_indices(n, m):
    """
    Generates all unique combinations of m indices chosen from a set of n indices (0 to n-1).

    Args:
        n (int): The total number of elements (or indices) available.
        m (int): The number of elements (or indices) to choose in each combination.

    Returns:
        list: A list of tuples, where each tuple represents a unique combination of m indices.
    """
    if not (0 <= m <= n):
        raise ValueError("m must be between 0 and n (inclusive).")

    # Generate indices from 0 to n-1
    indices = range(n)

    # Use itertools.combinations to get all combinations of m indices
    combinations_iterator = itertools.combinations(indices, m)

    # Convert the iterator to a list of tuples
    return list(combinations_iterator)




class SOLD: 
    """
    Class to deal with all the fucntionalities, given SOLD library specs, to create simulations and probability distributions 
    """
    def __init__(self, sold_mat_df): 
        """
        Expect a dataframe with amino acid single letter code in rows and positions (zero indexed) for columns
        """
        amino_acids = sold_mat_df.index.values.astype(str)
        positions = sold_mat_df.columns.astype(int)
        mat = sold_mat_df.to_numpy()
        parent = {}   
        mutation_probs = defaultdict(dict)  
        all_parent_probs = [] 
        for i,r in enumerate(positions):
            probs = mat[:,i] 
            if np.any(probs): 
                assert np.allclose(np.sum(probs), 1), "expected probabilities: check column " + np.str(r) 
                #check that these are probs 
                parent_prob = np.max(probs) 
                all_parent_probs.append(parent_prob) 
                parent_aa = amino_acids[np.argmax(probs)] 
                parent[r] = str(parent_aa)
                inds = np.flatnonzero(probs) 
                mutation_probs[r] = {str(amino_acids[i]):float(probs[i]) for i in inds} 
        self.mutation_probs = mutation_probs
        self.parent = parent
        parent_seq = ['N']*len(parent) 
        #making sure this is done robustly in case data entry is not in order
        parent_pos = list(np.sort(list(parent.keys())))
        for i, v in parent.items():
            parent_seq[parent_pos.index(i)] = v
        self.parent_seq = ''.join(parent_seq)  
        self.mut_positions = parent_pos 
        self.all_parent_probs = all_parent_probs
    def compute_prob_n_mutations(self, N): 
        """
        Compute the probability of seeing N mutations 
        """
        # number of positions 
        M = len(self.all_parent_probs)
        # get indices of mutations
        mut_probs = np.zeros(N) 
        all_parent_probs = np.copy(self.all_parent_probs) 
        for i in range(N): 
            # find which indices of the positions are mutated 
            mutated_indices = get_n_choose_m_indices(M, i) 
            total_mut_prob = 0
            for k in mutated_indices: 
                loc_all_probs = np.copy(all_parent_probs) 
                loc_all_probs[list(k)] = 1 - all_parent_probs[list(k)] 
                total_mut_prob += np.prod(loc_all_probs) 
            mut_probs[i] = total_mut_prob
        return mut_probs
        
    def generate_sequences(self, N): 
        """
        generate N sequences 
        Args:
            N: number of sequences to generate 
        """
        collector = np.empty((N,len(self.mut_positions)), dtype = str) 
        max_probs = [] 
        for pos, v in self.mutation_probs.items():
            choices = list(v.keys())
            probs = list(v.values())
            max_probs.append(max_probs) 
            index = self.mut_positions.index(pos) # find the index of the position to fill 
            loc_seq = np.random.choice(choices, N, p = probs) 
            collector[:, index] = loc_seq
        seqs = [''.join(s) for s in collector] 
        mutations = [hamming_dist(self.parent_seq, s) for s in seqs]   
        # compute the probability of mutations 
        return seqs, mutations

    def compute_hamming_distance(self, seqs, parent): 
        """
        Args: 
            seqs
            parent seq  
        """
        dist = [] 
        for s in seqs: 
            dist.append(hamming_dist(s, parent)) 
        return dist 


    