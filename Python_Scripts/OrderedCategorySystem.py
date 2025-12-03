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

def greedy_categorizer_softmax(best_syst, item_seq, D, treeLookup=None, temp=1):
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
        prob_dist = softmax(np.array(sys_scores), temp)
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

def compute_possible_scores(trials, order_dic, item_space):
    all_data = {t['P_ID']: [] for t in trials}
    D, item_hash = get_distance_mat(item_space)
    lookUpTree = defaultdict(lambda: None)
    for t in trials:
        d, l, o = t['DEPTH'], t['LOC'], t['ORDER']
        cat_assigns = t['ITEMS']
        if d == 2:
            syst = CategorySystem(item_hash, '..\\..\\Katie2025_AlienTaxonomist\\static_98863bd139ec98cf6bc52549beaaf679\\taxonomies\\tree2D.json')
        else:
            syst = CategorySystem(item_hash, '..\\..\\Katie2025_AlienTaxonomist\\static_98863bd139ec98cf6bc52549beaaf679\\taxonomies\\tree3D.json')
        orders = order_dic[f'{l}{o}']
        trial_data = []
        i = 0
        for i_ord in orders:
            start_syst = copy.deepcopy(syst)
            order_data = []
            for item in i_ord:
                syst_scores = []
                cats = []
                potential_systs = []
                current_syst = start_syst
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
                order_data.append((np.array(syst_scores), choice_idx))
                start_syst = potential_systs[choice_idx]  
            trial_data.append((order_data, i))
            i += 1
        all_data[t['P_ID']].append(trial_data)
    return all_data, D, lookUpTree
