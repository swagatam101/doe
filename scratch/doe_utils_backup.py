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

def create_synthetic_SOLD_matrix(num_mutated, length_of_protein, parent_prob = 0.85, mut_probs = [0.05, 0.05, 0.05]): 
    """
    Creates an artifical SOLD matrix 
    Args: 
        num_mutated: number of postions mutated 
        length_of_protein
        parent_prob: this makes all parent pos position uniform 
        mut_probs: how mane ranodm amino acdis to mutated to with probs 
    """
    parent = ''.join(np.random.choice(AMINO_ACIDS, length_of_protein))
    print("Parent protein:", parent)
    mutated_pos = np.sort(np.random.choice(range(length_of_protein), num_mutated, replace = False))
    print("Random mutaed positions", mutated_pos) 
    random_muts = [] 
    assert np.sum(mut_probs) + parent_prob == 1
    mut_dict = defaultdict(dict) 
    for i in mutated_pos: 
        draws = list(AMINO_ACIDS) 
        draws.remove(parent[i]) 
        to_draw = np.random.choice(draws, len(mut_probs), replace = False) 
        mut_dict[int(i)] = {parent[i]: parent_prob} 
        for k,l in enumerate(to_draw):
            mut_dict[int(i)].update({str(l): mut_probs[k]})
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
        
        self.amino_product = [''.join(x) for x in product(AMINO_ACIDS, AMINO_ACIDS)]
        self.pos_product = [np.asarray(x) for x in combinations(np.arange(self.mutated_region_length), 2)]

        # Now I need to create the codes for the amino acids, using Hadamard matrix, drom the first column which is constant 
        self.single_base_codes = H_20[:, 1:] # these are the codes for AMINO_ACIDS  
        self.pairwise_base_codes = np.kron(self.single_base_codes, self.single_base_codes) # for AMINO_ACID pairs
        
        # Notice that order in maintained in itertools.product and np.kron --- I checked--- this double_base_code_mat should now be assigned to the products 
        # This is truly how we genralize to other interaction combinations 
        # for ab in self.amino_product: 
        #     a = ab[0]
        #     b = ab[1]
        #     code_a = self.single_base_codes[AMINO_ACIDS.index(a), :]
        #     code_b = self.single_base_codes[AMINO_ACIDS.index(b), :]
        #     code_ab = self.pairwise_base_codes[self.amino_product.index(ab)]
        #     print(np.sum(code_ab - np.kron(code_a, code_b)))
        # Now I need to collect all of this in codes for every position that actually appear in the SOLD libraries, 
        # so I need to create positions by AMINO and postion_pairs by AMINO pairs matrices 
        
        independent_codes = []
        feature_names_independent = [] 
        for k,v in mutation_probs_variable_region_dict.items(): 
            for s,t in v.items(): 
                amino_index = AMINO_ACIDS.index(s)
                feature_names_independent.append(str(k) + '-' + str(s)) 
                independent_codes.append(self.single_base_codes[amino_index, :]) 

        pairwise_codes = [] 
        feature_names_pairwise = [] 
        for j, k in enumerate(self.pos_product): 
            for i, a in enumerate(self.amino_product): 
                if (a[0] in mutation_probs_variable_region_dict[k[0]]) &  (a[1] in mutation_probs_variable_region_dict[k[1]]): 
                    amino_index = self.amino_product.index(''.join(a)) 
                    feature_names_pairwise.append(str(k[0]) + '-' + str(a[0]) + ':' + str(k[1]) + '-' + str(a[1]))
                    pairwise_codes.append(self.pairwise_base_codes[amino_index, :])
        
        self.independent_codes = np.asarray(independent_codes) 
        #self.independent_codes = independent_codes #- np.mean(independent_codes, axis = 1)[:, np.newaxis]        
        self.shape_independent_weights  = np.shape(self.independent_codes)
        self.feature_names_independent = np.asarray(feature_names_independent) 
        self.feature_names_pairwise = np.asarray(feature_names_pairwise)
        self.pairwise_codes = np.asarray(pairwise_codes) 
        #self.pairwise_codes = pairwise_codes #- np.mean(pairwise_codes, axis = 1)[:, np.newaxis]
        self.shape_pairwise_weights  = np.shape(self.pairwise_codes) 
        self.independent_mapper = {i:j for i,j in zip(self.feature_names_independent, self.independent_codes)}
        self.pairwise_mapper = {i:j for i,j in zip(self.feature_names_pairwise, self.pairwise_codes)}


#######################################################################################################
class Sequence_encoder_simplex(Encoding_basics): 
    """
    Encodes sequence in simplex encodings --- this one doesn't try to create a general encoder but a SOLD library specifci encoder, which makes more sense anyway 
    Why encode variations / mutations you will never encounter? 
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
            
            local_code_I = np.zeros(self.shape_independent_weights)
            for i in range(self.mutated_region_length): 
                feature = str(i) + '-' + array_of_seq[i] 
                index = list(self.feature_names_independent).index(feature) 
                local_code_I[index, :] = self.independent_codes[index, :] 
            independent_codes.append(local_code_I)
            
            local_code_J = np.zeros(self.shape_pairwise_weights)
            for j, pos in enumerate(self.pos_product): 
                # need to find the index of amino_acid pairs 
                acid_pairs = array_of_seq[pos]
                feature = str(pos[0]) + '-' + acid_pairs[0] + ':' + str(pos[1]) + '-' + acid_pairs[1]
                index = list(self.feature_names_pairwise).index(feature)
                local_code_J[index, :] = self.pairwise_codes[index, :]
            pairwise_codes.append(local_code_J) 
            
        independent_codes = np.asarray(independent_codes)
        pairwise_codes = np.asarray(pairwise_codes)

        return independent_codes, pairwise_codes


#######################################################################################################

class Create_in_silico_model(Encoding_basics): 
    """
    Create an in silico model for simulation with independent and pairwise (epistatic) contributions 
    """
    def __init__(self, mutation_probs_variable_region_dict, independent_params = None, pairwise_params = None, baseline = 10): 
        """
        Args:
            mutation_probs ..  : pass the dict of mutation probs (this is generated by SOLD matrix class, attribute dict is called mutation_probs_variable_region_indexed) 
            MUST BE INDEXED BY THE POSITION OF THE MUTATED REGION, not the portein position... thsi is to make sure we can deal with different length proteins 
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
        I_defaults = {'rho':[0.2, 0.2], 'sparse_pdf_names': ['norm', 'norm'], 'noise_sigma' : 0.01, 'sparse_params': [{'loc': 1, 'scale': 0.2}, {'loc': -1, 'scale': 0.2}]} 
        #pairwise params pdf default ---
        P_defaults = {'rho':[0.2, 0.2], 'sparse_pdf_names': ['norm', 'norm'], 'noise_sigma' : 0.01, 'sparse_params': [{'loc': 0.75, 'scale': 0.2}, {'loc': -0.75, 'scale': 0.2}]} 

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
        samples_I, _ = self.Prob_I.samples(np.prod(self.shape_independent_weights))
        self.independent_weights = samples_I.reshape(self.shape_independent_weights)         
        self.Prob_P = Create_mixture(**self.pairwise_params)
        samples_P, _ = self.Prob_P.samples(np.prod(self.shape_pairwise_weights))
        self.pairwise_weights = samples_P.reshape(self.shape_pairwise_weights)              
        self.baseline = baseline    
    
    def model(self, independent_codes, pairwise_codes): 
        """
        Args: 
            independent_codes: the result of encoding my sequence encoder to independent codes --- these are tensors--- N seqs times A amino acids time L positions (shape_independet_weights) etc. 
            pairwise_codes: similar 
            masked: ignore the weights of independent and pairwise positions that are not variable! 
        """
        ans1 = np.einsum('ijk, jk -> i', independent_codes, self.independent_weights) 
        ans2 = np.einsum('ijk, jk -> i', pairwise_codes, self.pairwise_weights)
        return self.baseline + ans1 + ans2 
        
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
        plt.figure() 
        plt.imshow(self.independent_weights, vmin = -1, vmax = 1, cmap = 'RdBu', interpolation = 'None') 
        _  = plt.yticks(range(len(self.feature_names_independent)), self.feature_names_independent)
        _  = plt.xticks(range(self.shape_independent_weights[1]))
        plt.title("Independent weights") 
        plt.colorbar() 
        plt.figure(figsize = (5, 70))
        plt.imshow(self.pairwise_weights, vmin = -1, vmax = 1, cmap = 'RdBu', aspect = 'auto', interpolation = 'None') 
        _  = plt.yticks(range(len(self.feature_names_pairwise)), self.feature_names_pairwise)
        _  = plt.xticks(range(self.shape_pairwise_weights[1]))
        plt.title("Pairwise weights") 
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

        self.independent_mask, self.pairwise_mask, self.feature_names_independent, self.feature_names_pairwise = \
        create_masked_features(self.mutated_region_length, self.mutation_probs_variable_region_dict) 


    def fit(self, seqs, activities, lambda_I = 0.01, lambda_P = 0.1): 
        """
        Fit seqs to their activities 
        The seqs are ONLY variable regions seqs concatenated! No point trying to fit regions that don't vary in the SOLD experiment! 
        Args: 
            seqs
            activities: vector of real values 
        """
        assert len(seqs) == len(activities), "Seqs (X) and activities (y) should be same length vectors"
        self.encoder = sequence_encoder(self.mutated_region_length)
        I_encodings, P_encodings = self.encoder.encode_seqs(seqs)
        # Now I need to select the features that are actually explored in the SOLD matrix---both for independent and pairwise 
        # first fit the independent parameters so that the pairwise paramaters are truly only pariwise, and cannot be explained away by independent by reparameterization 
        
        self.features = np.asarray([np.concatenate((np.ravel(indt[self.independent_mask]), \
                                                    np.ravel(pair[self.pairwise_mask]))) for indt, pair in zip(I_encodings, P_encodings)]) 
        self.independent_indices = np.arange(len(self.feature_names_independent)) # first few are independent features 
        self.number_of_features = len(self.feature_names_independent) + len(self.feature_names_pairwise)
        self.pairwise_indices = np.arange(len(self.feature_names_independent), self.number_of_features)  # the second set is pariwise features 

        # I need to perform a constrained optimization

        self.constraints = _create_constraint_mat(self.feature_names_pairwise, self.feature_names_independent) 
        beta = cp.Variable(self.number_of_features)
        constraints = self.constraints @ beta == np.zeros(len(self.constraints)) 
        
        loss = cp.sum_squares(activities - self.features @ beta)    
        penalty = (lambda_I * cp.norm1(beta[self.independent_indices]) +
                   lambda_P * cp.norm1(beta[self.pairwise_indices]))
        objective = cp.Minimize(loss + penalty)
        # Define the problem and solve
        problem = cp.Problem(objective, [constraints])
        problem.solve()        
        predicted_activities = np.dot(self.features, beta.value) 
        return beta.value, predicted_activities  
        
    