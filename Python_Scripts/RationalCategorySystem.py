import numpy as np
import copy
from scipy.special import logsumexp
from scipy.stats import t
from collections import defaultdict

VERBOSE = False 

def get_tdist(x, a, mu, sigmasq, lambd):
    scale = np.sqrt(sigmasq * (1.0 + (1.0/lambd)))
    return t.logpdf(x, df=a, loc=mu, scale=scale)
class RationalModel:
    '''
    Implementation for continuous features only
    updated so can start with an already populated system 
    Adapted from: https://github.com/johnmcdonnell/Rational-Models-of-Categorization/blob/master/particle.py
    -- modified such that once participants make the maximum number of categories the coupling constraint goes to 1.0
    and no probability of creating a new category is calculated
    '''

    def __init__(self, c, mu_0, sigmasq_0, lambda_0, a_0, partition=None, stimuli=None, max_new_clusters=np.inf):
        if partition is None or stimuli is None:
            self.partition = []    
            self.stimuli = []
            self.N = 0
            self.clusters = 0
            self.label_to_cluster = defaultdict(lambda: None)
            self.cluster_to_label = defaultdict(lambda: None)
        else:
            self.initialize_from_existing(stimuli, partition)
        self.c = c # coupling parameter == probability items are from the same category 
        self.mu_0 = mu_0 # prior on feature means
        self.sigmasq_0 = sigmasq_0 # prior on feature variances 
        self.lambda_0 = lambda_0 # strength of mean prior
        self.a_0 = a_0 # strength of variance prior
        self.max_clusters = self.clusters + max_new_clusters

    def initialize_from_existing(self, stimuli, partition):
        if len(stimuli) != len(partition):
            raise ValueError("Stimuli and partition must be same length")
        self.stimuli = copy.deepcopy(stimuli)
        self.N = len(stimuli)
        unique_labels = sorted(set(partition), key=str)
        self.label_to_cluster = defaultdict(lambda:None, {label: i for i, label in enumerate(unique_labels)})
        self.cluster_to_label = defaultdict(lambda:None, {i: label for i, label in enumerate(unique_labels)})
        self.partition = [self.label_to_cluster[label] for label in partition]
        self.clusters = len(unique_labels)

    def find_distribution(self, k, i, lambda_i, a_i, n):
        """
        For a given cluster computes the current mean and variance.
        """
        items = [self.stimuli[idx][i] 
                 for idx, cl in enumerate(self.partition)
                 if cl == k]
        if n>0:
            xbar = np.mean(items)
        else:
            xbar = 0
        if n > 1:
            # POPULATION OR SAMPLE (ddof = 1 for sample mean)
            var = np.var(items, ddof=1)
        else:
            var = 0
        
        mu_i = ((self.lambda_0[i]*self.mu_0[i]) + (n * xbar)) / lambda_i
        if n == 0:
            sigmasq_i=self.sigmasq_0[i]
        else:
            sigmasq_i = ((self.a_0[i]*self.sigmasq_0[i]) + ((n-1.0) * var) +
                         ((self.lambda_0[i]*n)/lambda_i) * ((self.mu_0[i] - xbar)**2)) / a_i
        return mu_i, sigmasq_i

    def prob_density(self, k, i, x):
        if k == self.clusters:
            n = 0
        else:
            n = sum([item == k for item in self.partition])
        lambda_i = self.lambda_0[i] + n 
        a_i = self.a_0[i] + n
        mu_i, sigmasq_i = self.find_distribution(k, i, lambda_i, a_i, n)
        prob = get_tdist(x, a_i, mu_i, sigmasq_i, lambda_i)
        return prob

    def stimulus_prob(self, stim_idx, k):
        stim = self.stimuli[stim_idx]
        return sum(self.prob_density(k, i, stim[i]) for i in range(len(stim)))
    
    def compute_posterior(self, stim_idx):
        if self.clusters < self.max_clusters:
            pk  = np.empty(self.clusters+1)
            pfk = np.empty(self.clusters+1)
        else:
            pk  = np.empty(self.clusters)
            pfk = np.empty(self.clusters)
        
        for k in range(self.clusters):
            pk[k] = np.log(
                (self.c * sum([cl == k for cl in self.partition])) /
                ((1.0 - self.c) + (self.c * self.N))
            )
            pfk[k] = self.stimulus_prob(stim_idx, k)
        if self.clusters < self.max_clusters:
            pk[self.clusters] = np.log(
                (1.0 - self.c) / ((1.0 - self.c) + (self.c * self.N))
            )
            pfk[self.clusters] = self.stimulus_prob(stim_idx, self.clusters)
        num = pk + pfk
        denom = logsumexp(num)
        pkf = num - denom
        if VERBOSE:
            print("p(k): ", np.exp(pk), " sum=", np.sum(np.exp(pk)))
            print("p(f|k): ", np.exp(pfk))
            print("p(k|f): ", np.exp(pkf), " sum=", np.sum(np.exp(pkf)))
        self.current_posterior = pkf
        return pkf
    
    def get_item_likelihood(self, stim, label):
        # add stimulus to system
        self.stimuli.append(stim) 
        stim_idx = len(self.stimuli) - 1  
        # compute prob distribution
        posterior = self.compute_posterior(stim_idx)
        # get choice idx and update system
        if label not in self.label_to_cluster:
            choice_idx = self.clusters
            self.label_to_cluster[label] = choice_idx
            self.cluster_to_label[choice_idx] = label
            self.clusters += 1
            if self.clusters == self.max_clusters:
                self.c = 1.0  # force c to 1 if at max clusters
        else:
            choice_idx = self.label_to_cluster[label]
        self.partition.append(choice_idx)
        self.N += 1
        return posterior, choice_idx