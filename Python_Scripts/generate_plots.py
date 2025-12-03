import pandas as pd
from scipy import stats
from matplotlib import pyplot as plt

def order_effects_plot(df, fname='Figures/AXB_by_order_loc', figsize=(3.2, 4), legend=False):
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
            x = ['A', 'X', 'B']
            order = orders[j].lower()[0]
            loc = locations[i][0]
            y = grouped_df[(grouped_df['LOC'] == loc) & (grouped_df['ORDER'] == order)][['PROP_A', 'PROP_X', 'PROP_B']].values[0]
            y_err = sem[(sem['LOC'] == loc) & (sem['ORDER'] == order)][['PROP_A', 'PROP_X', 'PROP_B']].values[0]
            ax.bar(x, y, color=['#D81B60', '#1E88E5', '#FFC107'], edgecolor='black', width=0.75, linewidth=0.5)
            if i == 0:
                ax.set_title(f'{orders[j]}', fontsize=9.5)
                ax.set_xticks([])
            # elif i == 3:
            #     ax.set_xticks(['A', 'X', 'B'])
            # else:
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
                plt.Rectangle((0,0),0.5,0.5, facecolor='#D81B60', edgecolor='black', linewidth=0.5),
                plt.Rectangle((0,0),0.5,0.5, facecolor='#1E88E5', edgecolor='black', linewidth=0.5),
                plt.Rectangle((0,0),0.5,0.5, facecolor='#FFC107', edgecolor='black', linewidth=0.5),
            ]
            labels = ['A', 'X', 'B']
            fig.legend(
                handles,
                labels,
                loc='lower center',
                bbox_to_anchor=(0.22, -0.06),
                ncol=3,
                frameon=False,
                fontsize=9
            )


    fig.savefig(f'{fname}.jpg', dpi=400, bbox_inches='tight')
    
def new_category_plot(df, fname='Figures/X_by_depth', figsize=(3,3)):
    df_hyp3 = df[['DEPTH', 'HAS_X']].groupby(['DEPTH']).mean().reset_index()
    # sem = df[['DEPTH', 'HAS_X']].groupby(['DEPTH']).sem().reset_index()    
    # df_hyp3['SEM'] = sem['HAS_X']
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

    #  green = #004D40

def generate_summary_AXB_plot(df, column, colVals, names, figsize=(6.25, 2.15), overall=False, fname='Figures/summary'):
    stim_df = df.groupby([column]).mean(numeric_only=True).reset_index()[[column, 'PROP_A', 'PROP_B', 'PROP_X']]   
    counts = df.groupby([column]).count().reset_index()[[column, 'PROP_A', 'PROP_B', 'PROP_X']]   
    sem = df.groupby([column]).sem(numeric_only=True).reset_index()[[column, 'PROP_A', 'PROP_B', 'PROP_X']]  
    # condifence intervals instead??
    numAx = len(colVals)
    if overall:
        avg = df[['PROP_A', 'PROP_X', 'PROP_B']].mean(numeric_only=True).values
        overall_err = df[['PROP_A', 'PROP_X', 'PROP_B']].sem(numeric_only=True).values
        numAx += 1
    fig, axes = plt.subplots(1, numAx, figsize=figsize, constrained_layout=True)
    for i in range(numAx):
            ax = axes[i]
            x = ['A', 'X', 'B']
            if overall and i == 0:
                y = avg
                yerr = overall_err
                ax.text(-0.42, 0.89, f'T = {df.shape[0]}', size=8)
                ax.set_title('Overall', fontsize=9.5)
            elif overall:
                val = colVals[i-1]
                y = stim_df[stim_df[column] == val][['PROP_A', 'PROP_X', 'PROP_B']].values[0]
                yerr = sem[sem[column] == val][['PROP_A', 'PROP_X', 'PROP_B']].values[0]
                ax.text(-0.42, 0.89, f"T = {counts[counts[column] == val]['PROP_A'].iloc[0]}", size=8)
                ax.set_title(f'{names[i-1]}', fontsize=9.5)
            else:
                val = colVals[i]
                y = stim_df[stim_df[column] == val][['PROP_A', 'PROP_X', 'PROP_B']].values[0]
                yerr = sem[sem[column] == val][['PROP_A', 'PROP_X', 'PROP_B']].values[0]
                ax.text(-0.42, 0.89, f"T = {counts[counts[column] == val]['PROP_A'].iloc[0]}", size=8)
                ax.set_title(f'{names[i]}', fontsize=9.5)
            ax.bar(x, y, color=['#D81B60', '#1E88E5', '#FFC107'], edgecolor='black', width=0.75, linewidth=0.5)#, yerr=yerr, capsize=3)
            # ax.set_xticks(['A', 'X', 'B'])
            ax.set_xticks([])
            ax.tick_params(axis='both', which='major', labelsize=8)
            if i == 0:
                ax.set_yticks([0.0, 0.5, 1.0])
                ax.set_ylabel('Mean % of items', fontsize=9)
            else:
                ax.set_yticks([])
            ax.set_ylim(0, 1)

            handles = [
                plt.Rectangle((0,0),0.5,0.5, facecolor='#D81B60', edgecolor='black', linewidth=0.5),
                plt.Rectangle((0,0),0.5,0.5, facecolor='#1E88E5', edgecolor='black', linewidth=0.5),
                plt.Rectangle((0,0),0.5,0.5, facecolor='#FFC107', edgecolor='black', linewidth=0.5),
            ]
            labels = ['A', 'X', 'B']
            fig.legend(
                handles,
                labels,
                loc='lower right',
                bbox_transform=fig.transFigure,
                bbox_to_anchor=(1.02, -0.15),
                ncol=3,
                frameon=False,
                fontsize=8
            )
    fig.savefig(f'{fname}.jpg', dpi=300, bbox_inches='tight')


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
