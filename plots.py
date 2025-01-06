import pandas as pd

mofoc_preds = pd.read_csv('/kaggle/input/mofoc-preds/mofoc_preds.csv')
plays = pd.read_csv('/kaggle/input/nfl-big-data-bowl-2025/plays.csv')

preds_all = pd.merge(mofoc_preds, plays, on = ['gameId', 'playId'], how = 'left')

def count_matches(group):
    return sum(group['preds'] == group['true'])/len(group)

teams_d = preds_all.groupby(['defensiveTeam']).apply(count_matches)

leader = teams_d.to_frame(name = 'prop_correct').sort_values('prop_correct')




import seaborn as sns
import matplotlib.pyplot as plt

preds_some = preds_all[preds_all['preds'] != 2]

teams_o = preds_some.groupby(['possessionTeam', 'preds'])['expectedPointsAdded'].mean().reset_index()

teams_o = teams_o.pivot(index = 'possessionTeam', columns = 'preds', values = 'expectedPointsAdded').reset_index()


teams_o.columns = ['possessionTeam', 'openread_EPA', 'closedread_EPA']

readplot = sns.scatterplot(teams_o, x = 'openread_EPA', y = 'closedread_EPA')
readplot.plot([-0.25, 0.4], [-0.25, 0.4])
readplot.set(xlabel = 'Expected Points Added on MOFO-read plays', ylabel = 'Expected Points Added on MOFC-read plays')
plt.xlim(-0.3, 0.45)

plt.ylim(-0.55, 0.48)


for i, point in teams_o.iterrows():
    readplot.text(point['openread_EPA'] + 0.001, point['closedread_EPA'], str(point['possessionTeam']))



import pandas as pd
import os

os.chdir('..')

plays = pd.read_csv('plays_dat.csv')


plays.loc[(plays['gameId'] == 2022090800) & (plays['playId'] == 1657)]

