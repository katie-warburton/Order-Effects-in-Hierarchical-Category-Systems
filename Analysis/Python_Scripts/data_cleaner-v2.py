import json
import argparse
import pandas as pd

DISTRACTORS = ['1', '3.01', '29.01', '31']

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
    if cat_choices['1'][0] != 'A':
        incorrect += 1
    if cat_choices['3.01'][0] != 'A':
        incorrect += 1
    if cat_choices['31'][0] != 'B':
        incorrect += 1
    if cat_choices['29.01'][0] != 'B':
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

def extract_dfs(exp_data):
    trial_data = []
    participant_data = []
    sequence_data = []
    # trees = {}
    for pid, trials in exp_data.items():
        total_errors = 0
        if pid[0] == 'p':
            pool = 'prolific'
        elif pid[0] == 'r':
            pool = 'rep'
        else:
            pool = 'test'
        participant_row = {'P_ID': pid}
        for tr in trials[1:-2]:
            if tr['part'] == 'experiment':
                d, l, o = list(tr['condition'])
                trial_row = tr['category_choices']
                # if o.lower() in ['f', 'b', 'm']:
                #     sequence_row = {f't{i+1:02}': it for i, it in enumerate(tr['item_order'])}
                #     print(sequence_row)
                # else:
                #     sequence_row = {} # should it be order things are added to the tree??
                sequence_row = {f't{i+1:02}': it for i, it in enumerate(tr['item_order'])} # still not sure the best way to handle sequence for all at once??
                trial_row = {item: cat for pair in trial_row for item, cat in pair.items()}
                if None in trial_row.values():
                    trial_row = get_missing_label(trial_row, tr['final_tree'])
                errors = check_distractors(trial_row)
                trial_row['P_ID'] = pid
                trial_row['DEPTH'] = d
                trial_row['LOC'] = l
                trial_row['ORDER'] = o.lower()
                trial_row['STIMULI'] = tr['stimuli'][-1]
                trial_row['ERRORS'] = errors
                trial_row['POOL'] = pool
                trial_data.append(trial_row)

                sequence_row['P_ID'] = pid
                sequence_row['DEPTH'] = d
                sequence_row['LOC'] = l
                sequence_row['ORDER'] = o.lower()
                sequence_row['STIMULI'] = tr['stimuli'][-1]
                sequence_data.append(sequence_row)
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
    participant_data.to_csv('Analysis/Results/participant_data_test.csv', index=None)
    # make an save trial data in a data frame
    trial_data = pd.DataFrame.from_dict(trial_data)
    trial_data = trial_data[['P_ID', 'DEPTH', 'LOC', 'ORDER'] + [f'{i}' for i in range(9, 24)] + DISTRACTORS + ['ERRORS', 'STIMULI', 'POOL']]
    trial_data.rename(columns={f'{i}':f'I{i:02}' for i in range(9, 24)}, inplace=True)
    trial_data.rename(columns={'1': 'I01', '3.01': 'I03', '29.01': 'I29', '31': 'I31'}, inplace=True)
    trial_data.to_csv('Analysis/Results/trial_data_test.csv', index=None)
    #make and save sequence data frame
    sequence_data = pd.DataFrame.from_dict(sequence_data)
    sequence_data = sequence_data[['P_ID', 'DEPTH', 'LOC', 'ORDER'] + [f't{i:02}' for i in range(1, 14)] + ['STIMULI']]
    sequence_data.to_csv('Analysis/Results/sequence_data_test.csv', index=None)

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
       
def main(filepath, specific_ids=[]):
    with open(filepath, 'r') as f:
        data = json.load(f)
    # specific uid is used if errors in normal way of storing data 
    test = data[-1:]
    test = get_experimental_trials(test, 't')
    extract_dfs(test)   

main('Analysis/Results/results-98863bd139ec98cf6bc52549beaaf679-2025-12-01-04-10-08.json')
