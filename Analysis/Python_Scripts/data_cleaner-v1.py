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
    if cat_choices['1'][0] != 'A':
        incorrect += 1
    if cat_choices['3.01'][0] != 'A':
        incorrect += 1
    if cat_choices['31'][0] != 'B':
        incorrect += 1
    if cat_choices['29.01'][0] != 'B':
        incorrect += 1
    return incorrect

def extract_dfs(exp_data):
    trial_data = []
    participant_data = []
    for pid, trials in exp_data.items():
        total_errors = 0
        if pid[0] == 'p':
            pool = 'prolific'
        else:
            pool = 'rep'
        participant_row = {'P_ID': pid}
        for tr in trials[1:-2]:
            if tr['part'] == 'experiment':
                d, l, o = list(tr['condition'])
                trial_row = tr['category_choices']
                errors = check_distractors(trial_row)
                trial_row['P_ID'] = pid
                trial_row['DEPTH'] = d
                trial_row['LOC'] = l
                trial_row['ORDER'] = o.lower()
                trial_row['STIMULI'] = tr['stimuli'][-1]
                trial_data.append(trial_row)
                trial_row['ERRORS'] = errors
                trial_row['POOL'] = pool
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
    participant_data.to_csv('Order Effects/Analysis/Results/participant_data.csv', index=None)
    # make an save trial data in a data frame
    trial_data = pd.DataFrame.from_dict(trial_data)
    trial_data = trial_data[['P_ID', 'DEPTH', 'LOC', 'ORDER'] + [f'{i}' for i in range(9, 24)] + DISTRACTORS + ['ERRORS', 'STIMULI', 'POOL']]
    trial_data.rename(columns={f'{i}':f'I{i:02}' for i in range(9, 24)}, inplace=True)
    trial_data.rename(columns={'1': 'I01', '3.01': 'I03', '29.01': 'I29', '31': 'I31'}, inplace=True)
    trial_data.to_csv('Order Effects/Analysis/Results/trial_data.csv', index=None)
    #TO DO: make and save reaction time data frame (for potential future interest -- not current project)

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
    prolific = [d for d in data if ('prolific_id' in d.keys() and d['study_id'] == '6865dd825bc59eb3524941fb') 
                or (d['uid'] in specific_ids)]
    prolific= get_experimental_trials(prolific, 'p')
    rep = [d for d in data if 'REP' in d.keys()]
    rep = remove_duplicates(rep)
    rep = get_experimental_trials(rep, 'r')
    extract_dfs(prolific | rep)   

main('Order Effects/Analysis/Results/results-98863bd139ec98cf6bc52549beaaf679-2025-09-24-02-06-06.json', ['bdd4194403da271bff66e4e237429bdd'])
