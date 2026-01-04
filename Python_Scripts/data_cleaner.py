import json
import argparse
import pandas as pd
import OrderedCategorySystem as OCS

DISTRACTORS = ['1', '3.01', '29.01', '31']
D, ITEM_HASH = OCS.get_distance_mat([i for i in range(1, 32)])
ITEM_HASH[3.01] = 2
ITEM_HASH[29.01] = 28

def get_demographics(trial):
    demo = {'part': 'demographics'}
    if trial['gender'] == 'self_describe':
        demo['gender'] = trial['self_gender']
    elif trial['gender'] == 'prefer_not':
        demo['gender'] = ''
    else:
        demo['gender'] = trial['gender']
    demo['age'] = int(trial['age'])
    return demo

def get_experimental_trials(data, prefix):
    experiment_trials = {}
    i = 1
    for participant in data:
        attempts = 0
        pid = f'{prefix}{i:03d}'
        experiment_trials[pid] = []
        if prefix == 'rep':
            pass
        for trial in participant['results']:
            
            if 'part' in trial.keys() and trial['part'] == 'experiment':
                experiment_trials[pid].append(trial)
            elif 'part' in trial.keys() and trial['part'] == 'demographics':
                experiment_trials[pid].append(get_demographics(trial['response']))
            elif 'part' in trial.keys() and trial['part'] == 'sanity-check':
                attempts += 1
            elif 'total-minutes' in trial.keys():
                experiment_trials[pid].append({'minutes': trial['total-minutes']})
        experiment_trials[pid].append({"part":"check", "attempts": attempts})
        i += 1
    return experiment_trials

def check_distractors(cat_choices):
    incorrect = 0
    if cat_choices['1'][0] != 'L':
        incorrect += 1
    if cat_choices['3.01'][0] != 'L':
        incorrect += 1
    if cat_choices['31'][0] != 'R':
        incorrect += 1
    if cat_choices['29.01'][0] != 'R':
        incorrect += 1
    return incorrect

def find_in_tree(cat, item):
    if len(cat['children']) > 0:
        for child in cat['children']:
            label = find_in_tree(child, item)
            if label is not None:
                return label
    elif len(cat['items']) > 0:
        if float(item) in cat['items']:
            return cat['name']
    return None

def get_missing_label(trial_row, final_tree):
    items = [it for it, cat in trial_row.items() if cat is None]
    for it in items:
        label = find_in_tree(final_tree, it)
        trial_row[it] = label
    return trial_row

def replace_cat(cat_label):
    if cat_label == 'AA':
        return 'L1'
    elif cat_label == 'AB':
        return 'L2'
    elif cat_label == 'AY':
        return 'L3'
    elif cat_label == 'BY':
        return 'R3'
    elif cat_label == 'BA':
        return 'R2'
    elif cat_label == 'BB':
        return 'R1'
    elif cat_label == 'XY':
        return 'X1'
    elif cat_label == 'A':
        return 'L'
    elif cat_label == 'B':
        return 'R'
    else:
        return cat_label
    
def extract_dfs(exp_data):
    trial_data, participant_data, sequence_data = [], [], []
    for pid, trials in exp_data.items():
        if trials[0]['age'] < 18: # everyone should be under 18, but just in case
            continue
        total_errors = 0
        if pid[0:3] == 'p1_':
            pool = 'prolific1'
        elif pid[0:3] == 'p2_':
            pool = 'prolific2'
        elif pid[0:3] == 'r1_':
            pool = 'rep'
        else: # shouldn't get here with actual experimental data 
            pool = 'test'
        participant_row = {'P_ID': pid}
        for tr in trials[1:-2]:
            if tr['part'] == 'experiment':
                d, l, o = list(tr['condition'])
                trial_row = tr['category_choices']
                if pool == 'prolific2':
                    if o.lower() in ['f', 'b', 'm']:
                        sequence_row = {f't{i+1:02}': it for i, it in enumerate(tr['item_order'])}
                    else:
                        # currently order in which items are added to the tree -- could be the order in which their displayed too (would just be the same as the above row then)
                        sequence_row = {f't{i+1:02}': int(float(list(pair.keys())[0])) for  i, pair in enumerate(tr['category_choices']) }
                    # sequence_row = {f't{i+1:02}': it for i, it in enumerate(tr['item_order'])}
                    trial_row = {item: cat for pair in trial_row for item, cat in pair.items()}
                    sequence_row['P_ID'] = pid
                    sequence_row['DEPTH'] = d
                    sequence_row['LOC'] = l
                    sequence_row['ORDER'] = o.lower()
                    sequence_row['STIMULI'] = tr['stimuli'][-1]
                    sequence_data.append(sequence_row)
                if None in trial_row.values():
                    trial_row = get_missing_label(trial_row, tr['final_tree'])
                # change A and B to L and R
                trial_row = {it: replace_cat(cat) for it, cat in trial_row.items()}
                errors = check_distractors(trial_row)
                trial_row['P_ID'] = pid
                trial_row['DEPTH'] = d
                trial_row['LOC'] = l
                trial_row['ORDER'] = o.lower()
                trial_row['STIMULI'] = tr['stimuli'][-1]
                trial_row['ERRORS'] = errors
                trial_row['POOL'] = pool
                tree = OCS.CategorySystem(ITEM_HASH)
                tree.root = tree.parse_cats(tr['final_tree'])
                score = OCS.ordered_CKMM(tree.root, D)
                trial_row['SCORE'] = score
                trial_data.append(trial_row)
                total_errors += errors
        participant_row['GENDER'] = trials[0]['gender']
        participant_row['AGE'] = trials[0]['age']
        participant_row['ATTEMPTS'] = trials[-1]['attempts']
        participant_row['TOTAL_ERRORS'] = total_errors
        participant_row['MINUTES'] = trials[-2]['minutes']
        participant_row['POOL'] = pool
        participant_data.append(participant_row)
    # make and save participant data frame
    participant_data = pd.DataFrame.from_dict(participant_data)
    participant_data.to_csv('Results/participant_data.csv', index=None)
    # make an save trial data in a data frame
    trial_data = pd.DataFrame.from_dict(trial_data)
    trial_data = trial_data[['P_ID', 'DEPTH', 'LOC', 'ORDER'] + [f'{i}' for i in range(9, 24)] + DISTRACTORS + ['ERRORS', 'STIMULI', 'POOL', 'SCORE']]
    trial_data.rename(columns={f'{i}':f'I{i:02}' for i in range(9, 24)}, inplace=True)
    trial_data.rename(columns={'1': 'I01', '3.01': 'I03', '29.01': 'I29', '31': 'I31'}, inplace=True)
    trial_data.to_csv('Results/trial_data.csv', index=None)
    # make and save sequence data if included
    sequence_data = pd.DataFrame.from_dict(sequence_data)
    sequence_data = sequence_data[['P_ID', 'DEPTH', 'LOC', 'ORDER'] + [f't{i:02}' for i in range(1, 14)] + ['STIMULI']]
    sequence_data.to_csv('Results/sequence_data.csv', index=None)

def remove_duplicates(data):
    seen = set()
    duplicate_ids = {x['survey_code'] for x in data if x['survey_code'] in seen or seen.add(x['survey_code'])}
    no_dupes = []
    dupe_data = {d: [] for d in duplicate_ids}
    for d in data:
        if d['survey_code'] in duplicate_ids:
            dupe_data[d['survey_code']].append(d)
        else:
            no_dupes.append(d)
    for _, dupe in dupe_data.items():
        start = '2026-01-01T01:00:00'
        first_attempt = None
        for dat in dupe:
            d_start = dat['start']
            if d_start < start:
                start = d_start
                first_attempt = dat
        no_dupes.append(first_attempt)
    return no_dupes
       
def main(filepath):
    with open(filepath, 'r') as f:
        data = json.load(f)
    # specific uid is used if errors in normal way of storing data 
    prolific1 = [d for d in data if ('prolific_id' in d.keys() and d['study_id'] == '6865dd825bc59eb3524941fb') 
                or (d['uid'] in ['bdd4194403da271bff66e4e237429bdd'])]
    prolific1 = get_experimental_trials(prolific1, 'p1_')

    rep = [d for d in data if 'REP' in d.keys()]
    rep = remove_duplicates(rep)
    rep = get_experimental_trials(rep, 'r1_')

    prolific2 = [d for d in data if ('prolific_id' in d.keys() and d['study_id'] == '692d097af153eb01f86589b3') 
                or (d['uid'] in ['a818665ec602ae819875f02f1feb3d7a'])]
    prolific2 = get_experimental_trials(prolific2, 'p2_')

    extract_dfs(prolific1 | rep | prolific2)   

main('Results/results-98863bd139ec98cf6bc52549beaaf679-2025-12-07-23-58-07.json')
