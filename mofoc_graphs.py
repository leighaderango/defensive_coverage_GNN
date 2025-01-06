import pandas as pd
import numpy as np
import math
import itertools
import torch
from torch_geometric.data import Data
from torch_geometric.transforms import ToUndirected
import os
pd.set_option('future.no_silent_downcasting', True)

os.chdir('..')


# read and prepare separated data (plays and players)

graphs_dat = pd.read_csv('players_at_snap.csv')

graphs_dat['orig_position'] = graphs_dat['position']
graphs_dat = pd.get_dummies(graphs_dat, prefix = 'position', columns = ['position'])

# combine safeties to one indicator
graphs_dat['position_S'] = graphs_dat[['position_SS', 'position_FS']].max(axis = 1)


# keep only position indicator for safeties
position_columns = ['position_S']


graphs_dat['inMotionAtBallSnap'] = graphs_dat['inMotionAtBallSnap'].fillna(False)
    # ineligible players are left empty, so fill with false
graphs_dat['shiftSinceLineset'] = graphs_dat['shiftSinceLineset'].fillna(False)
    # ineligible players are left empty, so fill with false
graphs_dat['motionSinceLineset'] = graphs_dat['motionSinceLineset'].fillna(False)
    # ineligible players are left empty, so fill with false

# drop one play without playClockAtSnap and 23 without pff_passCoverage
graphs_dat = graphs_dat.dropna(subset = ['playClockAtSnap', 'pff_passCoverage'])


# replace true/false with 1/0
binary_features = ['inMotionAtBallSnap', 'shiftSinceLineset', 'motionSinceLineset', 'defender'] + position_columns

for feature in binary_features:
    graphs_dat[feature] = graphs_dat[feature].replace({True: 1, False: 0})


# convert height to inches
def height_to_in(str):
    split_str = str.split('-') 
    ft = int(split_str[0])
    inch = int(split_str[1])

    total = (ft*12) + inch
    return total

graphs_dat['height_in'] = graphs_dat['height'].apply(height_to_in)

# convert game clock to numeric (minutes)
def time_to_minutes(time_str):
    minutes, seconds = map(int, time_str.split(':'))
    total_minutes = minutes + (seconds/60)
    return total_minutes

graphs_dat['gameClockMins'] = graphs_dat['gameClock'].apply(time_to_minutes)


# convert coverage assignment to mofo/mofc/redzone classesd

def get_mof(orig_assignment):
    match orig_assignment: 
        case 'Cover-2' | 'Cover-6 Right' | 'Cover 6-Left' | 'Cover-0' | 'Quarters' | '2-Man' :
            return 'MOFO'

        case 'Cover-1' | 'Cover-1 Double' | 'Cover-3' | 'Cover-3 Seam' | 'Cover-3 Cloud Right' | 'Cover-3 Cloud Left' | 'Cover-3 Double Cloud' | 'Bracket':
            return 'MOFC'

        case _:
            return orig_assignment

graphs_dat['mof'] = graphs_dat['pff_passCoverage'].apply(get_mof)

# drop miscellaneous categories (dropping prevent, goal line, miscellaneous, leaving 9629 plays)
keep_forms = ['MOFO', 'MOFC', 'Red Zone']
graphs_dat = graphs_dat[graphs_dat['mof'].isin(keep_forms)]

# convert response to numbered index for input to model 
mof_to_id =  {label: idx for idx, label in enumerate(set(graphs_dat['mof']))}
graphs_dat['mofID'] = graphs_dat['mof'].map(mof_to_id)

print(mof_to_id)

plays_dat = graphs_dat.iloc[:, np.r_[0:50, 96, 97, 98]].drop('Unnamed: 0.1', axis = 1).drop_duplicates(ignore_index=True)
players_dat = graphs_dat.iloc[:, np.r_[1, 2, 52:98,]]

#plays_dat.to_csv('plays_dat.csv')
#players_dat.to_csv('players_dat.csv')

# iterate through grouped player dat df to create graph for each play, use plays_dat after
    # can imagine that iterating through plays_dat to find match to players_dat group id is quicker than iterating through all of player_dat to find all players on a given play
    # otherwise, would # iterate through list of plays (plays_dat) and get matching player data (from players_dat) to create graph

#one_play = graphs_dat.loc[(graphs_dat['gameId'] == 2022090800) & (graphs_dat['playId'] == 56)]
#second_play = graphs_dat.loc[(graphs_dat['gameId'] == 2022090800) & (graphs_dat['playId'] == 1967)]
#one_play.to_csv('buf_lar_220908_p1.csv')
#second_play.to_csv('buf_lar_220908_p2.csv')

# group df
grouped = graphs_dat.groupby(['gameId', 'playId'])
graphs_list = []


# full list of node features (with one-hot position columns dropped for dropped positions)
node_feature_cols = ['height_in', 'weight', 'o', 'dir', 's', 'motionSinceLineset', 'shiftSinceLineset', 'defender'] + position_columns


# drop linemen positions to minimize nodes
drop_positions = ['QB', 'C', 'T', 'G', 'DT', 'DE']
metadata_cols = ['mofID', 'yardlineNumber'] # + high safeties, added later


# iterate through plays
for play_num, node_df in grouped: 
    print(play_num)

    gameId = list(set(node_df['gameId']))[0]
    playId = list(set(node_df['playId']))[0]

    node_df = node_df[~node_df['orig_position'].isin(drop_positions)]

    # order by team and location
    node_df =  node_df.sort_values(['club', 'y'])


    # find graph-level info (response and metadata: yardline, quarter, down, playClockAtSnap
    play_row = plays_dat.loc[(plays_dat['gameId'] == gameId) & (plays_dat['playId'] == playId)]

    
    for _,row in play_row.iterrows():
        y_meta = row.loc[metadata_cols]
        los = float(row.loc['absoluteYardlineNumber'])
    

    # generate edge index and edge features
        # edge index: iterate through all pairwise combinations of 0:num players after conditioning
        # edge features have shape [num_edges, num_edge_features]

    remaining_verts = node_df.shape[0]

    edge_index = []
    edge_features = []

    for pair in itertools.combinations(range(0,remaining_verts), 2):

        p1 = node_df.iloc[pair[0],]
        p2 = node_df.iloc[pair[1],]

        # pick player furthest downfield (highest x) and furthest toward visitor sideline (highest y)
        if p1['x'] - p2['x'] < 0.5: # if players are less than 1/2 yard apart vertically,
            # then go by y
            if p2['y'] > p1['y']:
                # swap player indicies (to assist in keeping angles consistent)
                x = p1
                p1 = p2
                p2 = x 
        else: 
            #go by x
            if p2['x'] > p1['x']:
                # swap player indicies (to assist in keeping angles consistent)
                x = p1
                p1 = p2
                p2 = x 

        deltax = p2['x'] - p1['x']
        deltay = p2['y'] - p1['y']

        # find angle in radians and adjust to be on same scale as 'o' (0 deg points to visitor sideline)
        angle_rads = math.atan2(deltax, deltay) - math.pi/2

        angle_deg = math.degrees(angle_rads)

        if angle_deg < 0:
            angle_deg = angle_deg + 360


        # calculate pair's distance
        inter_dist =  math.sqrt((p2['x'] - p1['x'])**2 + (p2['y'] - p1['y'])**2)

        # if teammates
        teammates = 1 if p1['club'] == p2['club'] else 0


        edge_index.append(pair)
        edge_features.append([inter_dist, angle_deg, teammates])


    edge_index = torch.tensor(edge_index, dtype = torch.long).t().contiguous()
    edge_features = torch.tensor(edge_features, dtype = torch.float32)

    # generate node position matrix (player on-field locs)
        # and node feature tensor by iterating through player stats

    pos = []
    node_features_list = []
    high_safeties = 0

    for index, row in node_df.iterrows(): 
        x = row['x']
        y = row['y']


        # if in motion at ball snap is false, set speed = 0 and direction to orientation 
        if row['inMotionAtBallSnap'] == 0:
            row['s'] = 0
            row['dir'] = row['o']

        # count high safeties for use in metadata
        # if position_fs or position_ss is 1 AND player x is > 12 yards from LOS

        if (row['position_S'] == 1) and (abs(row['x'] - los) > 12):
            high_safeties += 1


        node_features = row.loc[node_feature_cols]

        node_features = pd.to_numeric(node_features, errors='coerce')
        node_features = node_features.fillna(0)

        node_features = node_features.to_numpy().reshape(1, -1)
        node_features_list.append(node_features)
        pos.append([x,y])


    pos = torch.tensor(pos, dtype = torch.float32)
    node_tensor = torch.tensor(np.vstack(node_features_list), dtype=torch.float32)

    # add num high safeties found from previous loop to meta
    y_meta = y_meta.to_list() + [high_safeties, gameId, playId]
    y = torch.tensor([y_meta], dtype = torch.long)

    # store as graph object
    play_graph = Data(x = node_tensor, edge_index = edge_index, edge_attr = edge_features, pos = pos, y = y)
    play_graph = ToUndirected()(play_graph)

    graphs_list.append(play_graph)


torch.save(graphs_list, 'mofo-mofc/mofoc_graphs.pt')
