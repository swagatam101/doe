"""
utils for DOE: major overhaul, Use Stormo simplex encoding (Hadamard matrix)  
"""

__copyright__ = "swagatam"
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
import random
from itertools import product, combinations
import scipy.stats as ss
import cvxpy as cp 
from sklearn.metrics import root_mean_squared_error
import re

# LOCAL 
AMINO_ACIDS = list(IUPACData.protein_letters)

#######################################################################################################

def create_synthetic_SOLD_matrix(num_mutated, length_of_protein, parent_prob = None, mut_probs = None): 
    """
    Creates an artifical SOLD matrix 
    Args: 
        num_mutated: number of postions mutated 
        length_of_protein
        parent_prob: this makes all parent pos position uniform 
        mut_probs: list of list of muated probs 
    """

    assert num_mutated == len(mut_probs), 'I need mutation probs as list of list of every position' 
    parent = ''.join(np.random.choice(AMINO_ACIDS, length_of_protein))
    print("Parent protein:", parent)
    mutated_pos = np.sort(np.random.choice(range(length_of_protein), num_mutated, replace = False))
    print("Random mutaed positions", mutated_pos) 
    random_muts = [] 
    for i, m in enumerate(mut_probs): 
        assert np.sum(m) + parent_prob[i] == 1
    mut_dict = defaultdict(dict) 
    for j,i in enumerate(mutated_pos): 
        draws = list(AMINO_ACIDS) 
        draws.remove(parent[i]) 
        to_draw = np.random.choice(draws, len(mut_probs[j]), replace = False) 
        mut_dict[int(i)] = {parent[i]: parent_prob[j]} 
        for k,l in enumerate(to_draw):
            mut_dict[int(i)].update({str(l): mut_probs[j][k]})
    sold_mat = np.zeros((len(AMINO_ACIDS), length_of_protein))
    for k,v in mut_dict.items(): 
        for base, prob in v.items(): 
            sold_mat[AMINO_ACIDS.index(base), k] = prob
    sold_mat_df = pd.DataFrame(sold_mat, index = AMINO_ACIDS, columns = np.arange(length_of_protein))
    return sold_mat_df, parent, mut_dict  


    
#######################################################################################################

def hamming_dist(seq1, seq2): 
    """
    compute hamming distance (number of mutations) for same length sequences 
    """
    return len(seq1)*hamming(list(seq1), list(seq2))

#######################################################################################################

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

#######################################################################################################

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
                mutation_probs[r] = {str(amino_acids[m]):float(probs[m]) for m in inds}

        self.mutation_probs = mutation_probs

        self.parent = parent
        parent_seq = ['N']*len(parent) 
        #making sure this is done robustly in case data entry is not in order
        parent_pos = list(np.sort(list(parent.keys()))) #NOTE this is SORTED
                
        mutation_probs_variable_region_indexed = defaultdict() 
        for i, p in enumerate(parent_pos): 
            mutation_probs_variable_region_indexed[i] = mutation_probs[p] 
        self.mutation_probs_variable_region_indexed = mutation_probs_variable_region_indexed

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

#######################################################################################################
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

class Sequence_encoder: 
    """
    Encodes sequence in single and pairiwise one hot codes 
    """
    def __init__(self, mutated_region_length): 
        """
        Args:
            mutated_region_length: the length of ONLY the variable region (mutated region) of the protein, NOT the whole protein length! 
        """
        binary = np.zeros(len(AMINO_ACIDS)) 
        self.mapper_independent = dict()
        for i,a in enumerate(AMINO_ACIDS): 
            loc_binary = np.copy(binary)
            loc_binary[i] = 1 
            self.mapper_independent[a] = loc_binary 

        self.amino_product = [''.join(x) for x in product(AMINO_ACIDS, AMINO_ACIDS)]
        self.pos_product = [np.asarray(x) for x in combinations(np.arange(mutated_region_length), 2)]
        self.mutated_region_length = mutated_region_length 
        
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
            assert len(seq) == self.mutated_region_length, "Protein length incorrect" 
            pairwise = np.zeros((len(self.amino_product), len(self.pos_product)))
            for j, pos in enumerate(self.pos_product): 
                # need to find the index of amino_acid pairs 
                acid_pairs = ''.join(array_of_seq[pos])
                idx = self.amino_product.index(acid_pairs) 
                pairwise[idx, j] = 1
            pairwise_codes.append(pairwise)
        return independent_codes, pairwise_codes

#######################################################################################################

class Create_mixture: 
    """
    Generates synthetic approximate "sparse" signal---
    this simply creates a mixture of distributions, one is close to zero (so, irrelevant and noise, so is Gaussian)
    and the other is any other distribution that is the "relevant" signal 
    Args: 
        rho: sparsity fractions (list) for the postive and negative components of weights 
        sparse_pdf_name: any pdf function from scipy.stats, pass name as string, 
        noise pdf is centered Gaussian here, meaning the component capturing approximate zero weights  
        noise_sigma: std. of zero centered noise 
        sparse_params: the params for the sparse signal pdf 
        
    Returns: 
        sparse signal pdf method 
        sprase signal rvs method to draw samples 
    """
    def __init__(self, rho = [0.4, 0.2], sparse_pdf_names = ['norm', 'norm'], \
                 noise_sigma = 0.02, sparse_params = [{'loc': 0.5, 'scale': 0.2}, {'loc': -0.5, 'scale': 0.2}]): 
        """
        See Args above 
        """
        assert np.sum(rho) < 1, 'Illegal probabilities for components, check rho' 
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
        ans[index == 0] = self.pdf1.rvs(size = size0)
        ans[index == 1] = self.pdf2.rvs(size = size1)
        ans[index == 2] = self.pdf3.rvs(size = size2)
        return ans, index         
    

#######################################################################################################
def create_code(bases): 
    """
    Create a code of length L -- one hot 
    """
    L = len(bases) 
    binary = np.zeros(L) 
    mapper = dict()
    for i,a in enumerate(bases): 
        loc_binary = np.copy(binary)
        loc_binary[i] = 1 
        mapper[a] = loc_binary 
    return mapper 

#######################################################################################################


class Encoding_basics: 
    """
    Base class used by both ecnoders and in silico model 
    """
    def __init__(self, mutation_probs_variable_region_dict): 
        """
        Args: 
            mutated region length: the number of positions mutated 
            mutation_probs_variable_region_dict: the standard way to encode the mutations --- see notes and guide 
        """
        self.mutated_region_length = len(mutation_probs_variable_region_dict)
        self.mutation_probs_variable_region_dict = mutation_probs_variable_region_dict

        # All mutations and encodings are relative to parent 
        # get product aminos and pos products 
        self.amino_product = [''.join(x) for x in product(AMINO_ACIDS, AMINO_ACIDS)]
        self.pos_product = [np.asarray(x) for x in combinations(np.arange(self.mutated_region_length), 2)]


        # now, for every position I need to encode the bases in a minimal code 
        # Create a code dict for every position 
        parent = ['']*len(mutation_probs_variable_region_dict) 
        self.independent_code_mapper = {} 
        self.positions = np.sort(list(self.mutation_probs_variable_region_dict.keys()))

        self.independent_positional_code_length = {} 
        for k, v in mutation_probs_variable_region_dict.items(): 
            code_length = len(v) - 1 
            parent_base = max(v, key=v.get)
            bases = [a for a in v.keys() if a!= parent_base] 
            mapper = create_code(bases) 
            mapper.update({parent_base: np.zeros(code_length)}) 
            indx = list(self.positions).index(k) 
            parent[indx] = parent_base
            self.independent_code_mapper[k] = mapper 
            self.independent_positional_code_length[k] = code_length

        self.independent_position_weights_name = defaultdict(list) 
        for k,v in self.independent_code_mapper.items(): 
            self.independent_position_weights_name[str(k)] = ['']*self.independent_positional_code_length[k]
            for s,t in v.items(): 
                if np.any(t): 
                    self.independent_position_weights_name[str(k)][np.flatnonzero(t)[0]] = s
                    
        # flattened names
        self.flattened_independent_position_weights_name = [] 
        for i in self.positions: 
            self.flattened_independent_position_weights_name.extend([str(i) +  '-' + a for a in self.independent_position_weights_name[str(i)]])
         
            
        self.parent = ''.join(parent)
        self.pos_product = [np.asarray(x) for x in combinations(self.positions, 2)] 
        # trying to keep this general... can handle positions that are not indexed by mutation position, but parent protein position, SHH! 

        # Now I have all the independent codes 
        #Lets go through all the positions are create the pairwise codes  

        code_size = defaultdict(int)
        pair_bases = defaultdict(list) 

        for i, j in self.pos_product: 
            for a, b in self.amino_product: 
                if (a in self.independent_code_mapper[i]) and (b in self.independent_code_mapper[j]): 
                    code_size[str(i) + ':' + str(j)] += 1 
                    pair_bases[str(i) + ':' + str(j)].append(a + b)

        self.pairwise_code_mapper = {}
        self.pairwise_positional_code_length = {} 
        
        for i, j in self.pos_product: 
            ind1 = list(self.positions).index(i)
            ind2 = list(self.positions).index(j)
            parent_pair = self.parent[ind1] + self.parent[ind2] 
            code_length = code_size[str(i) + ':' + str(j)] - 1 
            self.pairwise_positional_code_length[str(i) + ':' + str(j)] = code_length 
            bases = [a for a in pair_bases[str(i) + ':' + str(j)] if a!= parent_pair] 
            mapper = create_code(bases) 
            mapper.update({parent_pair: np.zeros(code_length)}) 
            self.pairwise_code_mapper[str(i) + ':' + str(j)] = mapper 


        # encode the parent sequence to get the code structure
        self.feature_names_independent = [] 
        self.encode_parent_independent = []
        for i, s in enumerate(self.positions): 
            self.encode_parent_independent.append(self.independent_code_mapper[s][self.parent[i]])

        self.feature_names_independent = [] 
        for s,v in self.independent_code_mapper.items():
            for k in v.keys(): 
                self.feature_names_independent.append(str(s) + '-' + k) 
                
        self.encode_parent_pairwise = []
        for i, j in self.pos_product: 
            ind1 = list(self.positions).index(i)
            ind2 = list(self.positions).index(j)
            a = parent[ind1]
            b = parent[ind2]
            self.encode_parent_pairwise.append(self.pairwise_code_mapper[str(i) + ':' + str(j)][str(a) + str(b)])    

        self.feature_names_pairwise = [] 
        for s,v in self.pairwise_code_mapper.items():
            for k in v.keys(): 
                self.feature_names_pairwise.append(str(s) + '-' + k) 

        # Need to create an address for all the pairwise weights 
        self.pair_position_weights_name = defaultdict(list) 
        for k,v in self.pairwise_code_mapper.items(): 
            self.pair_position_weights_name[k] = ['']*self.pairwise_positional_code_length[k]
            for s,t in v.items(): 
                if np.any(t): 
                    self.pair_position_weights_name[k][np.flatnonzero(t)[0]] = s
                    
        # flattened names

        self.flattened_pair_position_weights_name = [] 
        for i, j in self.pos_product: 
            ind1 = list(self.positions).index(i)
            ind2 = list(self.positions).index(j)
            self.flattened_pair_position_weights_name.extend([str(i) + ':' + str(j) + '-' + a[0] + ':' + a[1] for a in self.pair_position_weights_name[str(i) + ':' + str(j)]])
        
        self.code_length_pairwise = np.sum([len(code) for code in self.encode_parent_pairwise]) 
        self.flattened_pair_position_weights_name = np.asarray(self.flattened_pair_position_weights_name)
        self.code_length_independent = np.sum([len(code) for code in self.encode_parent_independent]) 
        self.number_of_features = self.code_length_independent + self.code_length_pairwise

        # find the constraint matrix \sum_{jb} J_ia, jb = 0 ... etc. 
        split_names = np.asarray([re.split(r'[:-]', s) for s in self.flattened_pair_position_weights_name])

        constraints = [] 
        for p in self.feature_names_independent:
            pos, base = p.split('-') 
            inds1 = np.flatnonzero((split_names[:, 0] == pos) & (split_names[:, 2] == base))
            inds2 = np.flatnonzero((split_names[:, 1] == pos) & (split_names[:, 3] == base))  
            v = np.zeros(self.code_length_pairwise)
            v[np.concatenate([inds1, inds2])] = 1 
            constraints.append(v)
        self.pairwise_constraints = np.asarray(constraints)
        

#######################################################################################################


class Sequence_encoder_simplex(Encoding_basics): 
    """
    This encodes in variable length codes, to get rid of null space, one hot simplex 
    """
    def __init__(self, mutation_probs_variable_region_dict): 
        """
        Args:
            mutated_region_length: the length of ONLY the variable region (mutated region) of the protein, NOT the whole protein length! 
        """
        super().__init__(mutation_probs_variable_region_dict) 
        
    def encode_seqs(self, protein_seqs): 
        """
        Args: 
            protein_seqs: this is only the variable region 
        """
        independent_codes = [] 
        pairwise_codes = [] 
        for seq in protein_seqs:
            array_of_seq = np.asarray(list(seq))
            assert len(seq) == self.mutated_region_length, "length mismatch of protein seq and attributes"
            
            local_code_I = [] 
            for i, s in enumerate(self.positions): 
                local_code_I.append(self.independent_code_mapper[s][array_of_seq[i]])
            independent_codes.append(local_code_I)   
            
            local_code_J = []
            for i, j in self.pos_product: 
                ind1 = list(self.positions).index(i)
                ind2 = list(self.positions).index(j)
                a = array_of_seq[ind1]
                b = array_of_seq[ind2]
                local_code_J.append(self.pairwise_code_mapper[str(i) + ':' + str(j)][str(a) + str(b)])     
            pairwise_codes.append(local_code_J) 

        flatten_independent = [] 
        for ind in independent_codes:
            flatten_independent.append([item for x in ind for item in x])
        flatten_independent = np.asarray(flatten_independent) 

        flatten_pairwise = [] 
        for ind in pairwise_codes:
            flatten_pairwise.append([item for x in ind for item in x])
        flatten_pairwise = np.asarray(flatten_pairwise) 
        
        return independent_codes, pairwise_codes, flatten_independent, flatten_pairwise


#######################################################################################################

def _fix_pairwise_weights(pairwise_weights, pairwise_constraints): 
    """
    Make sure J_{ia, jb} summed over ia or jb is zero so that it cannot be explained away by h_ia and h_jb --- 
    the pairwise weights cannot be explained by independent weights 
    """
    TOLERANCE = 1e-12
    MAXITER = 1000000
    num_iter = 0 
    
    delta = np.inf
    new_weights = np.copy(pairwise_weights)
    old_weights = np.copy(new_weights)
    while (delta > TOLERANCE) and (num_iter < MAXITER):
        num_iter += 1 
        for v in pairwise_constraints.astype(bool):
            new_weights[v] -= np.mean(new_weights[v]) 
        delta = root_mean_squared_error(old_weights, new_weights)
        old_weights = np.copy(new_weights)

    return new_weights, np.dot(pairwise_constraints, new_weights) 
        
class Create_in_silico_model(Encoding_basics): 
    """
    Create an in silico model for simulation with independent and pairwise contributions 
    """
    def __init__(self, mutation_probs_variable_region_dict, independent_params = None, pairwise_params = None): 
        """
        Args:
            mutation_probs ..  : pass the dict of mutation probs (this is generated by SOLD matrix class, attribute dict is called mutation_probs_variable_region_indexed) 
            MUST BE INDEXED BY THE POSITION OF THE MUTATED REGION, not the protein position... this is to make sure we can deal with different length proteins 
                example:
                {0: {'D': 0.05, 'K': 0.85, 'M': 0.05, 'Y': 0.05},
                 1: {'C': 0.05, 'G': 0.05, 'I': 0.05, 'P': 0.85},
                 2: {'F': 0.05, 'N': 0.05, 'R': 0.85, 'Y': 0.05},
                 3: {'G': 0.05, 'I': 0.05, 'L': 0.85, 'Q': 0.05},
                 4: {'A': 0.05, 'E': 0.05, 'R': 0.05, 'W': 0.85},
                 5: {'A': 0.05, 'D': 0.05, 'I': 0.85, 'K': 0.05}}
                where the keys are the index of the mutated regions and the values are dicts of probs of each amino acid
            idependent_params: {'rho':[0.2, 0.2], 'sparse_pdf_names': ['norm', 'norm'], 'noise_sigma' : 0.01, 'sparse_params': [{'loc': 1, 'scale': 0.2}, {'loc': -1, 'scale': 0.2}]} 
            pairwise_params: {'rho':[0.1, 0.1], 'sparse_pdf_names': ['norm', 'norm'], 'noise_sigma' : 0.01, 'sparse_params': [{'loc': 0.75, 'scale': 0.2}, {'loc': -0.75, 'scale': 0.2}]} 
            
        """
        # independent params pdf default --- 
        I_defaults = {'rho':[0.45, 0.45], 'sparse_pdf_names': ['norm', 'norm'], 'noise_sigma' : 0.01, 'sparse_params': [{'loc': 1, 'scale': 0.2}, {'loc': -1, 'scale': 0.2}]} 
        #pairwise params pdf default ---
        P_defaults = {'rho':[0.45, 0.45], 'sparse_pdf_names': ['norm', 'norm'], 'noise_sigma' : 0.01, 'sparse_params': [{'loc': 0.75, 'scale': 0.2}, {'loc': -0.75, 'scale': 0.2}]} 

        super().__init__(mutation_probs_variable_region_dict) 

        if independent_params is not None: 
            I_defaults.update(independent_params) 
        self.independent_params = I_defaults.copy()  

        if pairwise_params is not None: 
            P_defaults.update(pairwise_params) 
        # see the create mixture function
        self.pairwise_params = P_defaults.copy() 
        
        # Now I need to assign weights to individual and pairwise contributions 
        # first fill in the weights for the independent codes
        self.Prob_I = Create_mixture(**self.independent_params)
        self.independent_weights, _ = self.Prob_I.samples(self.code_length_independent)
        self.Prob_P = Create_mixture(**self.pairwise_params)
        pairwise_weights, _ = self.Prob_P.samples(self.code_length_pairwise)
        # fix the J_ia,jb problem -- ill-poses -> well-posed 
        self.pairwise_weights, _ = _fix_pairwise_weights(pairwise_weights, self.pairwise_constraints)
        
        
    def model(self, flatten_independent, flatten_pairwise = None): 
        """
        Args: 
            independent_codes: the result of encoding my sequence encoder to independent codes --- these are tensors--- \
            N seqs times A amino acids time L positions (shape_independet_weights) etc. 
            pairwise_codes: similar 
            masked: ignore the weights of independent and pairwise positions that are not variable! 
        """
        assert np.shape(flatten_independent)[1] == self.code_length_independent, "Code size mismatch" 
        
        ans1 = np.einsum('ij, j -> i', flatten_independent, self.independent_weights) 
        ans2 = 0 
        if flatten_pairwise is not None: 
            assert np.shape(flatten_pairwise)[1] == self.code_length_pairwise, "Code size mismatch" 
            ans2 = np.einsum('ij, j -> i', flatten_pairwise, self.pairwise_weights) 
        return ans1 + ans2 
        
    def plot_weights(self): 
        """
        Plotting functions 
        """
        # weights 
        I_samples = np.ravel(self.independent_weights) 
        GRID_SIZE = 1000
        x = np.linspace(np.min(I_samples), np.max(I_samples), GRID_SIZE)
        plt.figure() 
        plt.plot(x, self.Prob_I.pdf()(x))
        _ = plt.hist(I_samples, density=True, bins = 50)        
        plt.title('Distribution of independent weights') 

        P_samples = np.ravel(self.pairwise_weights) 
        x = np.linspace(np.min(P_samples), np.max(P_samples), GRID_SIZE)
        plt.figure() 
        plt.plot(x, self.Prob_P.pdf()(x))
        _ = plt.hist(P_samples, density=True, bins = 50)        
        plt.title('Distribution of pairwise weights') 
        #plt.colorbar

        
#######################################################################################################

def plot_encoding_independent(Encoder, code_mat, seq):
    """
    simple plotting to check 
    """
    plt.figure() 
    plt.imshow(code_mat, vmin = -1, vmax = 1, interpolation = 'None', aspect = 'auto', cmap = 'RdBu') 
    _ = plt.yticks(range(len(Encoder.feature_names_independent)), Encoder.feature_names_independent)
    _ = plt.xticks(range(Encoder.shape_independent_weights[1]))
    plt.title(seq)
    plt.xlabel('weights for simplex encoding') 

#######################################################################################################

def plot_encoding_pairwise(Encoder, code_mat, seq, figsize = (10,50)):
    """
    simple plotting to check 
    """
    plt.figure(figsize = figsize)  
    plt.imshow(code_mat, vmin = -1, vmax = 1, interpolation = 'None', aspect = 'auto', cmap = 'RdBu') 
    _ = plt.yticks(range(len(Encoder.feature_names_pairwise)), Encoder.feature_names_pairwise)
    _ = plt.xticks(range(Encoder.shape_pairwise_weights[1]))
    plt.title(seq)
    plt.xlabel('weights for simplex encoding') 
    
#######################################################################################################

class Fitting_model: 
    """
    l1 norm fitting model of data, with different penalty for indepednent and pairwise parameters 
    """
    def __init__(self, mutation_probs_variable_region_dict): 
        """
        I need to SOLD matrix dict to get to the right parameters to fit 
        
        Args: 
            pass the dict of mutation probs (this is generated by SOLD matrix class, attribute of dict is called mutation_probs_variable_region_indexed) 
            example:
            {0: {'D': 0.05, 'K': 0.85, 'M': 0.05, 'Y': 0.05},
             1: {'C': 0.05, 'G': 0.05, 'I': 0.05, 'P': 0.85},
             2: {'F': 0.05, 'N': 0.05, 'R': 0.85, 'Y': 0.05},
             3: {'G': 0.05, 'I': 0.05, 'L': 0.85, 'Q': 0.05},
             4: {'A': 0.05, 'E': 0.05, 'R': 0.05, 'W': 0.85},
             5: {'A': 0.05, 'D': 0.05, 'I': 0.85, 'K': 0.05}}
            where the keys are the index of the mutated regions and the values are dicts of probs of each amino acid
        
        """
        self.mutation_probs_variable_region_dict = mutation_probs_variable_region_dict
        self.mutated_region_length = len(mutation_probs_variable_region_dict) 
        self.encoder = Sequence_encoder_simplex(self.mutation_probs_variable_region_dict)

    def create_constraint_mat(self): 
        """
        Create the constraint mat for optimization 
        """
        temp = np.zeros((len(self.encoder.pairwise_constraints), self.encoder.code_length_independent))
        ans = np.hstack((temp, self.encoder.pairwise_constraints))
        return ans

    def fit(self, seqs, activities, lambda_I = 0.00001, lambda_P = 0.00001, fit = 'independent'): 
        """
        Fit seqs to their activities 
        The seqs are ONLY variable regions seqs concatenated! No point trying to fit regions that don't vary in the SOLD experiment! 
        Args: 
            seqs
            activities: vector of real values 
        """
        assert len(seqs) == len(activities), "Seqs (X) and activities (y) should be same length vectors"
        independent_codes, pairwise_codes, flatten_independent, flatten_pairwise = self.encoder.encode_seqs(seqs)

        # Now I need to select the features that are actually explored in the SOLD matrix---both for independent and pairwise 
        # first fit the independent parameters so that the pairwise paramaters are truly only pariwise, and cannot be explained away by independent by reparameterization 

        self.independent_indices = np.arange(self.encoder.code_length_independent) # first few are independent features 
        self.pairwise_indices = np.arange(self.encoder.code_length_independent, self.encoder.number_of_features)  # the second set is pariwise features 

        if fit == 'independent':
            self.features = flatten_independent
            beta = cp.Variable(self.encoder.code_length_independent)
            penalty = lambda_I * cp.norm1(beta)
            #Define the problem and solve
            loss = cp.sum_squares(activities - self.features @ beta)    
            objective = cp.Minimize(loss + penalty)
            problem = cp.Problem(objective)
            
        elif fit == 'both': 
            self.features = np.concatenate([flatten_independent, flatten_pairwise], axis = 1)
            beta = cp.Variable(self.encoder.number_of_features)
            penalty = (lambda_I * cp.norm1(beta[self.independent_indices]) + lambda_P * cp.norm1(beta[self.pairwise_indices]))
            constraints_mat = self.create_constraint_mat() 
            constraints = constraints_mat @ beta == np.zeros(len(constraints_mat)) 
            loss = cp.sum_squares(activities - self.features @ beta)    
            objective = cp.Minimize(loss + penalty)
            problem = cp.Problem(objective, [constraints])
        problem.solve()        
        predicted_activities = np.dot(self.features, beta.value) 
        return beta.value, predicted_activities
        
        # self.features = np.concatenate([flatten_independent, flatten_pairwise], axis = 1)
        
        # self.independent_indices = np.arange(self.encoder.code_length_independent) # first few are independent features 
        # self.pairwise_indices = np.arange(self.encoder.code_length_independent, self.encoder.number_of_features)  # the second set is pariwise features 

        # # I need to perform a constrained optimization
        # beta = cp.Variable(self.encoder.number_of_features)
        
        # loss = cp.sum_squares(activities - self.features @ beta)    
        # penalty = (lambda_I * cp.norm1(beta[self.independent_indices]) +
        #            lambda_P * cp.norm1(beta[self.pairwise_indices]))
        # objective = cp.Minimize(loss + penalty)
        # # Define the problem and solve
        # problem = cp.Problem(objective)
        # problem.solve()        
        # predicted_activities = np.dot(self.features, beta.value) 
        #return beta.value, predicted_activities  

        # self.features = np.concatenate([flatten_independent, flatten_pairwise], axis = 1)

        # beta_I = cp.Variable(self.encoder.code_length_independent)
        # loss = cp.sum_squares(activities - flatten_independent @ beta_I)    
        # penalty = lambda_I * cp.norm1(beta_I) 
        # objective = cp.Minimize(loss + penalty)
        # # Define the problem and solve
        # problem = cp.Problem(objective)
        # problem.solve()    
        # residual = activities - np.dot(flatten_independent, beta_I.value) 

        # beta_P = cp.Variable(self.encoder.code_length_pairwise)
        # loss = cp.sum_squares(residual - flatten_pairwise @ beta_P)    
        # penalty = lambda_P * cp.norm1(beta_P) 
        # objective = cp.Minimize(loss + penalty)
        # # Define the problem and solve
        # problem = cp.Problem(objective)
        # problem.solve()    

        