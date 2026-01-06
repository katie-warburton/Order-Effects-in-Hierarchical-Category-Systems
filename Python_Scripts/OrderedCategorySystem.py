'''
IMPLICIT ASSUMPTION: items in trees are ordered based on item_idx, categories are ordered 
'''
import json
import bisect
import copy
import random
import numpy as np
from collections import defaultdict
class Category:
    def __init__(self, label, depth=0, parent=None):
        self.label = label
        self.parent = parent
        self.depth = depth
        self.visible = True
        self.children = []
        self.item_idxs = []

    def __repr__(self):
        ret = "\t"*self.depth+repr(self.label)+"\n"
        for child in self.children:
            if child.visible:
                ret += child.__repr__()
        return ret
    
    def add_item(self, item, item_idx):
        loc = bisect.bisect(self.item_idxs, item_idx)
        self.item_idxs.insert(loc, item_idx)
        item_cat = Category(item, self.depth+1, self)
        item_cat.item_idxs = [item_idx]
        self.children.insert(loc, item_cat)
        parent = self.parent
        while parent != None:
            bisect.insort(parent.item_idxs, item_idx)
            parent = parent.parent

class CategorySystem:
    def __init__(self, item_hash, json_file=None):
        self.item_hash = item_hash
        self.num_nodes = 0
        self.num_items = 0
        self.can_add = []
        if json_file is None:
            self.root = None
        else:
            self.root = self.json_to_tree(json_file)
        
    def parse_cats(self, json_cat, depth=0, parent=None):
        cat = Category(json_cat['name'], depth=depth, parent=parent)
        cat.visible = json_cat['visible']
        if cat.visible:
            self.num_nodes += 1
        if len(json_cat['children']) == 0:
            self.can_add.append(cat)
            json_cat['items'].sort() # I assumed sorted, but just in case, because treating it as an ordered category system
            for item in json_cat['items']:
                cat.item_idxs.append(self.item_hash[item])
                item_cat = Category(item, cat.depth+1, self)
                item_cat.item_idxs.append(self.item_hash[item])
                cat.children.append(item_cat)
            self.num_items += len(json_cat['items'])
        for child in json_cat['children']:
            child_node = self.parse_cats(child, depth+1, cat)
            cat.children.append(child_node)
            cat.item_idxs += child_node.item_idxs
        return cat
    
    def json_to_tree(self, fp):
        with open(fp, 'r') as f:
            json_data = json.load(f)
        return self.parse_cats(json_data)

'''
CKMM Categorization Model
'''
def get_label(node):
    if len(node.children) == 0:
        return (tuple(node.item_idxs))
    
    return (tuple(get_label(child)
        for child in node.children
    ))

def ordered_CKMM(node, D, treeLookup=None):
    if treeLookup is None:
        treeLookup = defaultdict(lambda: None)
    label_key = get_label(node)
    cached = treeLookup[label_key]
    if cached is not None:
        return cached
    children = node.children
    k = len(children)
    if k == 0:
        score = 0
    elif k == 1:
        score = ordered_CKMM(children[0], D, treeLookup)
    elif k == 2:
        A, B = children
        a_score = ordered_CKMM(A, D, treeLookup)
        b_score = ordered_CKMM(B, D, treeLookup)
        d_AB = D[A.item_idxs, :][:, B.item_idxs]
        n_A = len(A.item_idxs)
        n_B = len(B.item_idxs)
        score = (d_AB.sum()*(n_A + n_B)) + a_score + b_score
    else:
        splits = k-1
        score = 0
        for i in range(splits):
            catsA, catsB = children[:i+1], children[i+1:]
            idx_A = [ix for cat in catsA for ix in cat.item_idxs]
            idx_B = [ix for cat in catsB for ix in cat.item_idxs]

            temp_node_A = Category('tempA')
            temp_node_A.children = catsA
            temp_node_A.item_idxs = idx_A
            a_score = ordered_CKMM(temp_node_A, D, treeLookup)

            temp_node_B = Category('tempB')
            temp_node_B.children = catsB
            temp_node_B.item_idxs = idx_B
            b_score = ordered_CKMM(temp_node_B, D, treeLookup)

            d_AB = D[idx_A, :][:, idx_B]
            n_A = len(idx_A)
            n_B = len(idx_B)
            score += (d_AB.sum()*(n_A + n_B)) + a_score + b_score
        score = score / splits
    treeLookup[label_key] = score
    return score 
            
def greedy_categorizer(best_syst, item_seq, D, treeLookup = None):
    cat_choices = {}
    if treeLookup is None:
        treeLookup = defaultdict(lambda: None)
    for item in item_seq:
        high_score = float('-inf')
        syst = best_syst
        potential_systs = []
        best_cat = []
        for i in range(len(syst.can_add)):
            test_syst = copy.deepcopy(syst)
            cat = test_syst.can_add[i]
            cat.add_item(item, best_syst.item_hash[item])
            test_syst.num_items += 1
            if not cat.visible:
                cat.visible = True
                test_syst.num_nodes += 1
            score = ordered_CKMM(test_syst.root, D, treeLookup)
            if score > high_score: 
                high_score = score
                potential_systs = [test_syst]
                best_cat = [cat.label]
            elif score == high_score:
                potential_systs.append(test_syst)
                best_cat.append(cat.label)
        best_idx = random.randint(0, len(potential_systs) - 1)
        best_syst = potential_systs[best_idx]
        cat_choices[item] = best_cat[best_idx]
    return best_syst, cat_choices    

def softmax(x, temp):
    x = x / temp
    x = x - np.max(x, axis=-1, keepdims=True)
    e_x = np.exp(x)
    return e_x / np.sum(e_x, axis=-1, keepdims=True)

def greedy_categorizer_softmax(best_syst, item_seq, D, treeLookup=None, temp=1, alpha=0):
    cat_choices = {}
    if treeLookup is None:
        treeLookup = defaultdict(lambda: None)
    for item in item_seq:
        syst = best_syst
        potential_systs = []
        sys_scores = []
        cats = []
        for i in range(len(syst.can_add)):
            test_syst = copy.deepcopy(syst)
            cat = test_syst.can_add[i]
            cat.add_item(item, best_syst.item_hash[item])
            test_syst.num_items += 1
            if not cat.visible:
                cat.visible = True
                test_syst.num_nodes += 1
            sys_scores.append(ordered_CKMM(test_syst.root, D, treeLookup))
            potential_systs.append(test_syst)
            cats.append(cat.label)
        choice_prob =  softmax(np.array(sys_scores), temp)
        random = 1/len(sys_scores)
        prob_dist = ((1-alpha) * choice_prob) + (alpha*random)
        best_idx = np.random.choice([ix for ix in range(len(potential_systs))], p=prob_dist)
        best_syst = potential_systs[best_idx]
        cat_choices[item] = cats[best_idx]
    return best_syst, cat_choices

def get_distance_mat(items, min_it=None, max_it=None, noise=0):
    item_hash = {lab+1: lab for lab in range(len(items))}
    if min_it is None:
        min_it = np.min(items)
    if max_it is None:
        max_it = np.max(items)
    item_values = (np.array([items]) - min_it) / (max_it - min_it)
    items_T = item_values.T 
    D = np.abs((item_values - items_T))
    if noise != 0:
        max_val = np.max(D)
        sigma = noise * max_val
        noise_vec = np.random.normal(0, sigma, size=D.shape)
        D_noise = (noise_vec + noise_vec.T) / 2
        D = D + D_noise
        np.fill_diagonal(D, 0)
    return D, item_hash


def precompute_possible_scores(trials, item_space, folder='Python_Scripts/BaseSystems'):
    D, item_hash = get_distance_mat(item_space)
    lookUpTree = defaultdict(lambda: None)
    all_data = {tr['P_ID']: [] for tr in trials}
    for tr in trials:
        d = tr['DEPTH']
        cat_assigns = tr['ITEMS']
        if d == 2:
            syst = CategorySystem(item_hash, f'{folder}/tree2D.json')
        else:
            syst = CategorySystem(item_hash, f'{folder}/tree3D.json')
        trial_data = []
        for t in range(len(cat_assigns)):
            syst_scores = []
            cats = []
            potential_systs = []
            current_syst = syst
            item = tr['SEQUENCE'][f't{t+1:02}']
            for i in range(len(current_syst.can_add)):
                test_syst = copy.deepcopy(current_syst)
                cat = test_syst.can_add[i]
                cat.add_item(item, current_syst.item_hash[item])
                test_syst.num_items += 1
                if not cat.visible:
                    cat.visible = True
                    test_syst.num_nodes += 1
                score = ordered_CKMM(test_syst.root, D, lookUpTree)
                syst_scores.append(score)
                cats.append(cat.label)
                potential_systs.append(test_syst)
            cat_choice = cat_assigns[f'I{item:02}']
            choice_idx = cats.index(cat_choice)
            syst = potential_systs[choice_idx]  
            trial_data.append((syst_scores, choice_idx))
        all_data[tr['P_ID']].append(trial_data)
    return all_data, D, lookUpTree


def get_participant_loglike(data, determ=False, temp=1.0, alpha=0.0):
    total_log_like = 0.0
    for trial in data:
        sys_scores_array = np.array([s for s, _ in trial], dtype=np.float64)
        choice_indices = np.array([c for _, c in trial], dtype=int)
        if determ:
            probs = np.zeros_like(sys_scores_array)
            probs[np.arange(len(sys_scores_array)), sys_scores_array.argmax(1)] = 1
            probs = probs/probs.sum(axis=1)[:,None] # make options equally likely if the same score
        else:
            probs = softmax(sys_scores_array, temp)
        K = probs.shape[1] 
        choice_prob = probs[np.arange(probs.shape[0]), choice_indices]
        # random_baseline = np.full_like(choice_prob, 1.0 / K)
        # might want to come up with a way to fit it to two-level probabilities
    
        p_choice = ((1 - alpha) * choice_prob) + (alpha * (1.0 / K))
        total_log_like += np.sum(np.log(p_choice))
    return total_log_like

# Assumes you've precomputed participant system scores
def get_total_log_like(participants, determ=False, temp=1.0, alpha=0.0):
    total_log_like = 0.0
    for data in participants.values():
        total_log_like += get_participant_loglike(data, determ, temp, alpha)
    return total_log_like

def find_best_params(data, params, determ=False):
    best_ll = -np.inf
    if determ:
        alphas = params
        best_a = None
        for a in alphas:
            ll = get_total_log_like(data, determ=True, alpha=a)
            if ll > best_ll:
                best_ll = ll
                best_a = a
        stats = (best_a, best_ll)
    else:
        temps, alphas = params
        best_t, best_a = None, None
        for t in temps:
            prev_ll = None
            for a in alphas:
                ll = get_total_log_like(data, determ, t, a)
                if ll > best_ll:
                    best_ll = ll
                    best_t, best_a = t, a
                # Alphas are concave - can stop if it starts decreasing
                if prev_ll is not None and ll < prev_ll:
                    break
                prev_ll = ll
        stats = (best_t, best_a, best_ll)
    return stats