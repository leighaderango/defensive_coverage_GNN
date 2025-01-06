import pandas as pd
import numpy as np

plays = pd.read_csv('data/plays.csv')

# get passing plays with pff_passCoverage (list of plays) from plays
passing_plays = plays[~pd.isnull(plays['passResult'])]
    # 9736 passing plays
        # 23 missing pff_passCoverage


# get positions at snap from tracking_week_x in one dataframe
def get_tracking_atsnap(weeks = range(1, 10)):

    tracking_at_snap = pd.DataFrame()

    for week in weeks:
        filename = 'data/tracking_week_' + str(week) + '.csv'
        tracking_wk= pd.read_csv(filename)

        tracking_snaps_wk = tracking_wk[tracking_wk['frameType'] == 'SNAP']
            # select only frames at snap

        tracking_at_snap = pd.concat([tracking_snaps_wk, tracking_at_snap], ignore_index = True)
        print(f'week {week}')

    return(tracking_at_snap)


#tracking_at_snap = get_tracking_atsnap()
tracking_at_snap = pd.read_csv('tracking_at_snap.csv')
tracking_at_snap = pd.merge(passing_plays, tracking_at_snap, on = ['gameId', 'playId'], how = 'left')
    # keep only passing plays


# match tracking to player-level info (ex. position) from players
players = pd.read_csv('data/players.csv')
players = players[['nflId', 'height', 'weight', 'position']]

players_at_snap = pd.merge(tracking_at_snap, players, on = 'nflId', how = 'left')


# get 'inMotionAtBallSnap","shiftSinceLineset","motionSinceLineset' from player_play
player_play = pd.read_csv('data/player_play.csv')
player_play = player_play[['gameId', 'playId', 'nflId', 'inMotionAtBallSnap', 'shiftSinceLineset', 'motionSinceLineset']]

players_at_snap = pd.merge(players_at_snap, player_play, on = ['gameId', 'playId', 'nflId'])

players_at_snap['defender'] = np.where(players_at_snap['club'] == players_at_snap['defensiveTeam'], 1, 0)  


players_at_snap.to_csv('players_at_snap.csv')

