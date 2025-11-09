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
    left_df = df[df['LOC'] == loc]
    items = np.array(items)
    seq_dfs = []
    for lab, first_idx, other_idx in orders:
       first = items[first_idx]
       other = items[other_idx].tolist()
       item_subset = left_df[left_df['ORDER'] == lab]
       same_as_first = item_subset[[first] + other].apply(lambda x: x[other] == x[first], axis=1)*1
       same_as_first['PROP_SAME'] = same_as_first[other].sum(axis=1) / 8
       subseq_df = pd.concat([item_subset[['P_ID','POOL', 'DEPTH', 'LOC', 'ORDER']], item_subset[first], same_as_first], axis=1)
       subseq_df.rename(columns={first: 'FIRST'}, inplace=True)
       subseq_df.rename(columns={other[i] : f'OTH_{i+1}' for i in range(len(other))}, inplace=True)
       seq_dfs.append(subseq_df)
    return pd.concat(seq_dfs)

def get_seq_data(df, locs, orders):
    dfs_by_loc = []
    for label, items in locs:
        seq_df = get_seq_df(df, items, label, orders)
        dfs_by_loc.append(seq_df)
    seq_df = pd.concat(dfs_by_loc)
    return seq_df.reset_index().drop('index', axis=1)

def get_jsds(df, loc):
    df_at_loc = df[df['LOC'] == loc]
    dist_a = np.array(df_at_loc[df_at_loc['ORDER'] == 'a'][['PROP_A', 'PROP_X', 'PROP_B']].mean().values)
    dist_f = np.array(df_at_loc[df_at_loc['ORDER'] == 'f'][['PROP_A', 'PROP_X', 'PROP_B']].mean().values)
    dist_m = np.array(df_at_loc[df_at_loc['ORDER'] == 'm'][['PROP_A', 'PROP_X', 'PROP_B']].mean().values)
    dist_b = np.array(df_at_loc[df_at_loc['ORDER'] == 'b'][['PROP_A', 'PROP_X', 'PROP_B']].mean().values)

    jsd_f = distance.jensenshannon(dist_f, dist_a)
    jsd_m = distance.jensenshannon(dist_m, dist_a)
    jsd_b = distance.jensenshannon(dist_b, dist_a)
    return jsd_f, jsd_m, jsd_b