"""
utils for DOE 
"""

__copyright__ = " "
__status__ = "Dev"
__doc__ = """ utilities """
__author__  = "Swagatam Mukhopadhyay"

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from Bio.Data import IUPACData
from collections import defaultdict
from scipy.spatial.distance import hamming
import itertools
from itertools import product, combinations

AMINO_ACIDS = list(IUPACData.protein_letters)



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

############################ code already in seqeunce encoder, but these are the algos ################

def one_hot_encode(protein_seqs):
    """
    Args:
        List of protein seqs 
    returns:
        20 x length of protein for one hot encodings
    """
    binary = np.zeros(len(AMINO_ACIDS)) 
    mapper = dict()
    for i,a in enumerate(AMINO_ACIDS): 
        loc_binary = np.copy(binary)
        loc_binary[i] = 1 
        mapper[a] = loc_binary 
    
    M = []
    for seq in protein_seqs:
        temp = np.asarray([mapper[k] for k in seq]).T
        M.append(temp)
    return np.asarray(M)


def pairwise_encode(protein_seq): 
    """
    pairwise encode protein sequences --- so 400 x l(l-1)/2 code, where l is the length of the protein 
    Args: 
        The protein sequence in the mutated positions, concatenated, not the whole sequence with constatnt region
    """
    L = len(protein_seq) # NOTE: I am generally only working with the subsequence that is mutated---not the whole sequence 
    array_of_seq = np.asarray(list(protein_seq))
    amino_product = [''.join(x) for x in product(AMINO_ACIDS, AMINO_ACIDS)]
    pos_product = [np.asarray(x) for x in combinations(np.arange(L), 2)]
    
    codes = np.zeros((len(amino_product), len(pos_product)))
    for j, pos in enumerate(pos_product): 
        # need to find the index of amino_acid pairs 
        acid_pairs = ''.join(array_of_seq[pos])
        idx = amino_product.index(acid_pairs) 
        codes[idx, j] = 1

    return codes 

#######################################################################################################

class sequence_encoder: 
    """
    Encodes sequence in single and pairiwise one hot codes 
    """
    def __init__(self, protein_length): 
        """
        Args:
            protein_length: L of the protein (mutated) region only 
        """
        binary = np.zeros(len(AMINO_ACIDS)) 
        self.mapper_independent = dict()
        for i,a in enumerate(AMINO_ACIDS): 
            loc_binary = np.copy(binary)
            loc_binary[i] = 1 
            self.mapper_independent[a] = loc_binary 

        self.amino_product = [''.join(x) for x in product(AMINO_ACIDS, AMINO_ACIDS)]
        self.pos_product = [np.asarray(x) for x in combinations(np.arange(protein_length), 2)]
        self.protein_length = protein_length 
        
    def encode_seqs(self, protein_seqs): 
        """
        Args: 
            protein_seqs
        """
        M = []
        for seq in protein_seqs:
            temp = np.asarray([self.mapper_independent[k] for k in seq]).T
            M.append(temp)
        independent_codes = np.asarray(M)

        # now compute the pairwise codes 
        pairwise_codes = [] 

        for seq in protein_seqs:    
            array_of_seq = np.asarray(list(seq))
            assert len(seq) == self.protein_length, "Protein length incorrect" 
            pairwise = np.zeros((len(self.amino_product), len(self.pos_product)))
            for j, pos in enumerate(self.pos_product): 
                # need to find the index of amino_acid pairs 
                acid_pairs = ''.join(array_of_seq[pos])
                idx = self.amino_product.index(acid_pairs) 
                pairwise[idx, j] = 1
            pairwise_codes.append(pairwise)
        return independent_codes, pairwise_codes

#######################################################################################################

class create_mixture: 
    """
    Generates synthetic approximate "sparse" signal---this simply creates a mixture of distributions, one is close to zero (so, irrelevant and noise, so is Gaussian)
    and the other is any other distribution that is the "relevant" signal 
    Args: 
        rho: sparsity fraction for the postive and negative components of weights 
        sparse_pdf_name: any pdf function from scipy.stats, pass name as string
        noise_sigma: std. of zero centered noise 
        sparse_params: the params for the sparse signal pdf 
        
    Returns: 
        sparse signal pdf method 
        sprase signal rvs method to draw samples 
    """
    def __init__(self, rho = [0.4, 0.2], sparse_pdf_names = ['norm', 'norm'], noise_sigma = 0.02, sparse_params = [{'loc': 0.5, 'scale': 0.2}, {'loc': -0.5, 'scale': 0.2}]): 
        """
        See Args above 
        """
        assert np.sum(rho) < 1, 'No zero component' 
        self.pdf1 = ss.norm(loc = 0, scale = noise_sigma)
        self.pdf2 = getattr(ss, sparse_pdf_names[0])(**sparse_params[0])
        self.pdf3 = getattr(ss, sparse_pdf_names[1])(**sparse_params[1])

        self.rho = rho 
        self.sparse_pdf_name = sparse_pdf_names
        self.noise_sigma = noise_sigma
        self.sparse_params = sparse_params
        
    def pdf(self): 
        """
        pdf(x) --- x is the support you want to evaluate the function on 
        """
        mixture = lambda x: (1-np.sum(self.rho))*self.pdf1.pdf(x) + self.rho[0]*self.pdf2.pdf(x) + self.rho[1]*self.pdf3.pdf(x)
        return mixture 
    
    def samples(self, N): 
        """
        draw N samples 
        """
        ans = np.zeros(N)
        index = np.asarray(random.choices([0,1,2], weights=[1-np.sum(self.rho), self.rho[0], self.rho[1]],k=N))
        size0 = np.sum(index == 0)
        size1 = np.sum(index == 1)
        size2 = np.sum(index == 2)
        print(size0, size1, size2)
        ans[index == 0] = self.pdf1.rvs(size = size0)
        ans[index == 1] = self.pdf2.rvs(size = size1)
        ans[index == 2] = self.pdf3.rvs(size = size2)
        return ans, index         
    

