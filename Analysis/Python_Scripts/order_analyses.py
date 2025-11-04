import copy
import pandas as pd
import numpy as np
from scipy.spatial import distance

def first_char(val):
    if val is not np.nan:
        return val[0]
    else:
        return np.nan

def get_level2_cat_stats(df_orig, items):
    df = copy.deepcopy(df_orig)
    df[items] = df[items].map(first_char)
    df['NUM_A'] = df[df[items] == 'A'].count(axis=1)
    df['NUM_B'] = df[df[items] == 'B'].count(axis=1)
    df['NUM_X'] = df[df[items] == 'X'].count(axis=1)
    df['PROP_A'] = df['NUM_A']/9
    df['PROP_B'] = df['NUM_B']/9
    df['PROP_X'] = df['NUM_X']/9
    df['HAS_X'] = df['NUM_X'].apply(lambda x: 1 if x > 0 else 0)
    return df

def get_seq_df(df, items, loc, orders):
    loc_df = df[df['LOC'] == loc]
    items = np.array(items)
    seq_dfs = []
    long_dfs = []
    for lab, first_idx, other_idx in orders:
       first = items[first_idx]
       other = items[other_idx].tolist()
       item_subset = loc_df[loc_df['ORDER'] == lab]
       same_as_first = item_subset[[first] + other].apply(lambda x: x[other] == x[first], axis=1)*1
       same_as_first['PROP_SAME'] = same_as_first[other].sum(axis=1) / 8
       same_as_first['FIRST_IT'] = first
       subseq_df = pd.concat([item_subset[['P_ID', 'STIMULI', 'POOL', 'DEPTH', 'LOC', 'ORDER']], item_subset[first], same_as_first], axis=1)
       subseq_df.rename(columns={first: 'FIRST'}, inplace=True)
       long_seq = subseq_df.melt(id_vars = ['P_ID', 'STIMULI', 'POOL', 'DEPTH', 'LOC', 'ORDER', 'FIRST', 'FIRST_IT'], value_vars=other, var_name='ITEM', value_name='SAME')
       long_seq.rename(columns={first: 'FIRST_CAT'}, inplace=True)
       subseq_df.rename(columns={other[i] : f'OTH_{i+1}' for i in range(len(other))}, inplace=True)
       seq_dfs.append(subseq_df)
       long_dfs.append(long_seq)
    return pd.concat(seq_dfs), pd.concat(long_dfs)

def get_seq_data(df, locs, orders):
    dfs_by_loc = []
    long_by_loc = []
    for label, items in locs:
        seq_df, long_df = get_seq_df(df, items, label, orders)
        dfs_by_loc.append(seq_df)
        long_by_loc.append(long_df)
    seq_df = pd.concat(dfs_by_loc)
    long_df = pd.concat(long_by_loc)
    return seq_df.reset_index().drop('index', axis=1), long_df.reset_index().drop('index', axis=1)

def jsd_no_nan(dist1, dist2):
    with np.errstate(invalid='ignore'):
        dist = distance.jensenshannon(dist1, dist2)
        if np.isnan(dist):
            dist = 0.0
    return dist

def get_jsds(df, loc):
    df_at_loc = df[df['LOC'] == loc]
    dist_a = np.array(df_at_loc[df_at_loc['ORDER'] == 'a'][['PROP_A', 'PROP_X', 'PROP_B']].mean().values)
    dist_f = np.array(df_at_loc[df_at_loc['ORDER'] == 'f'][['PROP_A', 'PROP_X', 'PROP_B']].mean().values)
    dist_m = np.array(df_at_loc[df_at_loc['ORDER'] == 'm'][['PROP_A', 'PROP_X', 'PROP_B']].mean().values)
    dist_b = np.array(df_at_loc[df_at_loc['ORDER'] == 'b'][['PROP_A', 'PROP_X', 'PROP_B']].mean().values)

    jsd_f = jsd_no_nan(dist_f, dist_a)
    jsd_m = jsd_no_nan(dist_m, dist_a)
    jsd_b = jsd_no_nan(dist_b, dist_a)
    return jsd_f, jsd_m, jsd_b

def perm_test(df, loc, it):
    jsd_f, jsd_m, jsd_b = get_jsds(df, loc)
    if loc == 'L':
        diff1 = jsd_b - jsd_f
        diff2 = jsd_b - jsd_m
    elif loc == 'R':
        diff1 = jsd_f - jsd_b
        diff2 = jsd_f - jsd_m
    seq_df = copy.deepcopy(df)
    seq_df = seq_df[(seq_df['LOC'] == loc) & (seq_df['ORDER'] != 'a')].reset_index()
    p1, p2 = 0, 0
    for _ in range(it):
        seq_df['ORDER'] = seq_df['ORDER'].sample(frac=1, random_state=13).values
        all_at_once = df[df['ORDER'] == 'a']
        test_df = pd.concat([seq_df, all_at_once])
        jsd_f, jsd_m, jsd_b = get_jsds(test_df, loc)
        if loc == 'L':
            temp_diff1 = jsd_b - jsd_f
            temp_diff2 = jsd_b - jsd_m
        elif loc == 'R':
            temp_diff1 = jsd_f - jsd_b
            temp_diff2 = jsd_f - jsd_m
        if temp_diff1 >= diff1:
            p1 += 1
        if temp_diff2 >= diff2:
            p2 += 1
    return p1/it, p2/it
