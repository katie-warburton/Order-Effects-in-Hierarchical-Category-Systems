import pandas as pd
import numpy as np
from scipy import stats
from matplotlib import pyplot as plt

def bootstrap_LXR_confidence_intervals(df, p):
    np.random.seed(13)
    avg_prop = np.empty((p, 3))
    N = df.shape[0]
    lb, ub = int(0.025*p)-1, int(0.975*p)-1
    df = df[['PROP_L', 'PROP_X', 'PROP_R']]
    for i in range(p):
        sample_df = df.sample(N, replace=True)
        avg_prop[i,:] = sample_df.mean(axis=0).to_list()
    l_vec, x_vec, r_vec = avg_prop[:,0], avg_prop[:,1], avg_prop[:,2]
    l_vec.sort(), x_vec.sort(), r_vec.sort()
    return np.array([l_vec[lb], x_vec[lb], r_vec[lb]]), np.array([l_vec[ub], x_vec[ub], r_vec[ub]])

def order_effects_plot(df, fname='Figures/LXR_by_order_loc', figsize=(3.2, 4), legend=False):
    grouped_df = df.groupby(['LOC', 'ORDER']).mean(numeric_only=True).reset_index()
    counts = df.groupby(['LOC', 'ORDER']).count().reset_index()
    sem = df.groupby(['LOC', 'ORDER']).sem(numeric_only=True).reset_index() 
    overall = df.groupby(['ORDER']).mean(numeric_only=True).reset_index()
    overall['LOC'] = ['O', 'O', 'O', 'O']

    overall_sem = df.groupby(['ORDER']).sem(numeric_only=True).reset_index()
    overall_sem['LOC']  = ['O', 'O', 'O', 'O']
    grouped_df = pd.concat([grouped_df, overall])
    sem = pd.concat([sem, overall_sem])
    fig, axes = plt.subplots(4, 4, figsize=figsize, constrained_layout=True)
    orders = ['Forward', 'Middle', 'Backward', 'All at once']
    locations = ['Left', 'Center', 'Right', 'Overall']
    for i in range(4):
        for j in range(4):
            ax = axes[i, j]
            ax.tick_params(axis='both', which='major', labelsize=9)
            x = ['L', 'X', 'R']
            order = orders[j].lower()[0]
            loc = locations[i][0]
            y = grouped_df[(grouped_df['LOC'] == loc) & (grouped_df['ORDER'] == order)][['PROP_L', 'PROP_X', 'PROP_R']].values[0]

            if i < 3:
                ci_lb, ci_ub  = bootstrap_LXR_confidence_intervals(df[(df['LOC'] == loc) & (df['ORDER'] == order)], 5000)
            else:
                ci_lb, ci_ub = bootstrap_LXR_confidence_intervals(df[df['ORDER'] == order], 5000)
            ci_lb = y - ci_lb
            ci_ub = ci_ub - y
            y_err = np.vstack([ci_lb, ci_ub])
            ax.bar(x, y, color=['#D81B60', '#1E88E5', '#FFC107'], edgecolor='black', width=0.75, linewidth=0.75, yerr=y_err, capsize=2, error_kw=dict(elinewidth=0.75, capthick=0.75))
            if i == 0:
                ax.set_title(f'{orders[j]}', fontsize=9.5)
                ax.set_xticks([])
            ax.set_xticks([])
            if j == 0:
                ax.set_ylabel(f'{locations[i]}', fontsize=10)
                ax.set_yticks([0.0, 0.5, 1.0])
            else:
                ax.set_yticks([])
            ax.set_ylim(0, 1)
            if i != 3:
                ax.text(-0.42, 0.86, f"T = {counts[(counts['LOC'] == loc) & (counts['ORDER'] == order)]['P_ID'].iloc[0]}", size=8)
            else:
                total_trials = counts.groupby('ORDER').sum().reset_index()
                ax.text(-0.42, 0.86, f"T = {total_trials[total_trials['ORDER'] == order]['P_ID'].iloc[0]}", size=8)
        if legend:
            handles = [
                plt.Rectangle((0,0),0.5,0.5, facecolor='#D81B60', edgecolor='black', linewidth=0.75),
                plt.Rectangle((0,0),0.5,0.5, facecolor='#1E88E5', edgecolor='black', linewidth=0.75),
                plt.Rectangle((0,0),0.5,0.5, facecolor='#FFC107', edgecolor='black', linewidth=0.75),
            ]
            labels = ['L', 'X', 'R']
            fig.legend(
                handles,
                labels,
                loc='lower center',
                bbox_to_anchor=(0.22, -0.06),
                ncol=3,
                frameon=False,
                fontsize=9
            )
    fig.savefig(f'{fname}.jpg', dpi=600, bbox_inches='tight')
    
def new_category_plot(df, fname='Figures/X_by_depth', figsize=(3,3)):
    df_hyp3 = df[['DEPTH', 'HAS_X']].groupby(['DEPTH']).mean().reset_index()
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    ax.bar(x='DEPTH', height='HAS_X' , data=df_hyp3, color=['#785ef0', '#fe6100'], edgecolor="black", capsize=6)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.0, 0.25, 0.5, 0.75, 1.0])
    ax.set_ylabel('% with X')
    ax.set_xlabel('Depth')
    ax.set_xticks([2, 3])
    ax.text(1.6, 0.92, f'T = {df.shape[0]}', size=10)
    ax.spines[['right', 'top']].set_visible(False)
    fig.savefig(f'{fname}.jpg', dpi=300, bbox_inches='tight') 

def generate_summary_LXR_plot(df, column, colVals, names, figsize=(6.25, 2.15), overall=False, fname='Figures/summary', legend=True, ylabel='Mean % of items'):
    stim_df = df.groupby([column]).mean(numeric_only=True).reset_index()[[column, 'PROP_L', 'PROP_R', 'PROP_X']]   
    counts = df.groupby([column]).count().reset_index()[[column, 'PROP_L', 'PROP_R', 'PROP_X']]   
    numAx = len(colVals)
    if overall:
        avg = df[['PROP_L', 'PROP_X', 'PROP_R']].mean(numeric_only=True).values
        numAx += 1
    fig, axes = plt.subplots(1, numAx, figsize=figsize, constrained_layout=True)
    for i in range(numAx):
            ax = axes[i]
            x = ['L', 'X', 'R']
            if overall and i == 0:
                y = avg
                ax.text(-0.42, 0.89, f'T = {df.shape[0]}', size=8)
                ax.set_title('Overall', fontsize=9.5)
                ci_lb, ci_ub  = bootstrap_LXR_confidence_intervals(df, 5000)
            elif overall:
                val = colVals[i-1]
                y = stim_df[stim_df[column] == val][['PROP_L', 'PROP_X', 'PROP_R']].values[0]
                ax.text(-0.42, 0.89, f"T = {counts[counts[column] == val]['PROP_L'].iloc[0]}", size=8)
                ax.set_title(f'{names[i-1]}', fontsize=9.5)
                ci_lb, ci_ub  = bootstrap_LXR_confidence_intervals(df[df[column] == val], 5000)
            else:
                val = colVals[i]
                y = stim_df[stim_df[column] == val][['PROP_L', 'PROP_X', 'PROP_R']].values[0]
                ax.text(-0.42, 0.89, f"T = {counts[counts[column] == val]['PROP_L'].iloc[0]}", size=8)
                ax.set_title(f'{names[i]}', fontsize=9.5)
                ci_lb, ci_ub  = bootstrap_LXR_confidence_intervals(df[df[column] == val], 5000)
            ci_lb = y - ci_lb
            ci_ub = ci_ub - y
            y_err = np.vstack([ci_lb, ci_ub])
            ax.bar(x, y, color=['#D81B60', '#1E88E5', '#FFC107'], edgecolor='black', width=0.75, linewidth=0.75, yerr=y_err, capsize=2,  error_kw=dict(elinewidth=0.75, capthick=0.75))
            ax.set_xticks([])
            ax.tick_params(axis='both', which='major', labelsize=8)
            if i == 0:
                ax.set_yticks([0.0, 0.5, 1.0])
                ax.set_ylabel(ylabel, fontsize=9)
            else:
                ax.set_yticks([])
            ax.set_ylim(0, 1)

            if legend:
                handles = [
                    plt.Rectangle((0,0),0.5,0.5, facecolor='#D81B60', edgecolor='black', linewidth=0.75),
                    plt.Rectangle((0,0),0.5,0.5, facecolor='#1E88E5', edgecolor='black', linewidth=0.75),
                    plt.Rectangle((0,0),0.5,0.5, facecolor='#FFC107', edgecolor='black', linewidth=0.75),
                ]
                labels = ['L', 'X', 'R']
                fig.legend(
                    handles,
                    labels,
                    loc='lower left',
                    bbox_transform=fig.transFigure,
                    bbox_to_anchor=(-0.02, -0.15),
                    ncol=3,
                    frameon=False,
                    fontsize=8
                )
    fig.savefig(f'{fname}.jpg', dpi=600, bbox_inches='tight')


def same_as_dist(seq_df, fname='Figures/same_as_dist', figsize=(4, 2.5)):
    dist = seq_df['PROP_SAME'].value_counts().reset_index()
    dist['prop'] = dist['count'] / seq_df.shape[0]
    max_prob = dist['prop'].max()
    labels =  dist['PROP_SAME'].to_list()
    labels.sort()

    # mean_prob = seq_df['PROP_SAME'].mean()
    # median_prob = seq_df['PROP_SAME'].median()

    fig, ax = plt.subplots(1, 1, figsize=figsize)
    ax.bar(x='PROP_SAME', height='prop', data=dist,  color='#004D40', edgecolor="black", width=1/(dist.shape[0]))
    # ax.axvline(x=mean_prob, color='black', linestyle='--')
    # ax.axvline(x=median_prob, color='black', linestyle='-')
    ax.set_xticks(labels, [0, 1, 2, 3, 4, 5, 6, 7, 8])
    ax.text(-0.05, 0.46, f'T = {seq_df.shape[0]}', fontsize=10)
    ax.set_ylabel('% of Trials (T)', fontsize=11)
    ax.set_xlabel('Number of Items', fontsize=11)
    ax.set_ylim(0, 0.5)
    ax.tick_params(axis='both', which='major', labelsize=10)
    ax.spines[['right', 'top']].set_visible(False)
    fig.tight_layout()
    fig.savefig(f'{fname}.jpg', dpi=300, bbox_inches='tight')
