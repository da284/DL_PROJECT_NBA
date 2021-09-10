"""
Things to mix and match:
	labels
	point differentials
	changing parameters (games back, max_diff of labels)
	more/less games in the database
	more basic counting stats (points, assists, blocks, steals)
"""


import pandas as pd
import numpy as np
from nba_api.stats.library import parameters
from nba_api.stats.static import players
from nba_api.stats.static import teams
from nba_api.stats.endpoints import leaguegamefinder
from nba_api.stats.endpoints import cumestatsplayer
from nba_api.stats.endpoints import teamgamelog
from nba_api.stats.library.parameters import SeasonAll
import time


FIRST_SEASON = 2004
GAMES_BACK = 8
NUM_FEATURES = 8
MAX_DIFF = 25
TEST_SPLIT = 0.1

NORMAL = True
WIN_PCT = False
CONVOLUTION = False
WEIGHTED = False

GAME_TABLE = pd.read_csv('data_tables/team_data_clean_' + str(FIRST_SEASON) + '.csv', sep=',')	
TEAM_TABLE = teams.get_teams()

def calc_label(home_pts, away_pts):
	return max(min(1.0, (home_pts - away_pts + MAX_DIFF)/(2 * MAX_DIFF)),0.0)

def get_labels():
	new_df = pd.DataFrame(GAME_TABLE, columns=['Game_ID', 'MATCHUP', 'PTS']).sort_values(['Game_ID'])
	d = {}
	for i in range(0,len(new_df),2):
		if '@' in new_df.iloc[i]['MATCHUP']:
			away_pts, home_pts = new_df.iloc[i]['PTS'], new_df.iloc[i+1]['PTS']
		else:
			home_pts, away_pts = new_df.iloc[i]['PTS'], new_df.iloc[i+1]['PTS']
		d[new_df.iloc[i]['Game_ID']] = calc_label(home_pts, away_pts)
	return d


def win_loss_convert(record):
	"""
	Accepts:
	record - list of 'W' and 'L'

	Returns:
	corresponding list of 0 and 1
	"""
	ret = []
	for game in record:
		if game == 'W':
			ret += [1]
		else:
			ret += [0]

	return ret

def get_home_win_pct(prev_games, home):
	"""
	Accepts:
	prev_games- dataframe of games before the game in question
	home- boolean that states whether or not the game is a home game

	Returns:
	Win percentage for that season either away or home
	Number of games this is based on
	"""

	record = prev_games['WL'].tolist()
	last_game =  prev_games.iloc[0]['DATE_CONV']
	if last_game % 1000 > 700:
		season_start = last_game - (last_game % 10000) + 700 
	else:
		season_start = last_game - 10000 - (last_game % 10000) + 700

	season_games = prev_games[prev_games['DATE_CONV'] > season_start]

	win_loss = []
	games = 0
	for i in range(len(season_games)):
		if home and ('@' not in season_games.iloc[i]['MATCHUP']):
			games += 1
			if season_games.iloc[i]['WL'] == 'W':
				win_loss += [1]
			else:
				win_loss += [0]

		if not home and ('@' in season_games.iloc[i]['MATCHUP']):
			games += 1
			if season_games.iloc[i]['WL'] == 'W':
				win_loss += [1]
			else:
				win_loss += [0]

	if len(win_loss) == 0:
		return 0, 0

	return (sum(win_loss) / len(win_loss)), games

def get_opponent_win_pct(prev_games):
	"""
	Accepts:
	prev_games- list of games to analyze

	Returns:
	win_pcts- list of win percentages of the opponents for that season in order
	point_diff- list of point differentials for these games
	"""

	win_pcts = []
	point_diff = []
	efgs = []
	team_id = prev_games.iloc[0]['Team_ID']
	for i in range(len(prev_games)):
		game = prev_games.iloc[i]
		game_id = game['Game_ID']
		opp_game = GAME_TABLE[(GAME_TABLE['Game_ID'] == game_id) & (GAME_TABLE['Team_ID'] != team_id)]
		win_pcts += [opp_game.iloc[0]['W_PCT']]
		efgs += [opp_game.iloc[0]['EFG']]

		point_diff += [game['PTS'] - opp_game.iloc[0]['PTS']]

	return win_pcts, point_diff, efgs

def win_pct_convert(pct, games):
	return (0.5 + games*pct)/(games + 1)


def get_team_stats(game, home):
	"""
	Accepts:
	game- a row describing the current game (including team)

	Returns:
	Array of needed stats for the team in the input row
	"""

	#games of this team that took place before the current one
	prev_games = GAME_TABLE[(GAME_TABLE['Team_ID'] == game['Team_ID']) & (GAME_TABLE['DATE_CONV'] < game['DATE_CONV'])].head(GAMES_BACK)

	#games do not exist for some reason
	if len(prev_games) < GAMES_BACK:
		return None

	#labels of previous games
	labels = [LABELS[game_id] for game_id in prev_games['Game_ID']]

	stats = []
	win_pct = prev_games.iloc[0]['W_PCT']

	# inputs for Convolution Network
	if CONVOLUTION:
		# padding for convolution
		stats += [win_pct_convert(win_pct, len(prev_games))] * GAMES_BACK

		stats += win_loss_convert(prev_games['WL'].tolist())
		stats += prev_games['FTM'].tolist()
		stats += prev_games['FTA'].tolist()
		stats += prev_games['OREB'].tolist()
		stats += prev_games['DREB'].tolist()
		stats += prev_games['TOV'].tolist()
		stats += prev_games['EFG'].tolist()
		stats += prev_games['STL'].tolist()
		stats += prev_games['BLK'].tolist()
		stats += labels
		stats += prev_games['AST'].tolist()
 

	# win percentage approach
	if WIN_PCT:
		opp_win_pct, point_diff, opp_efg = get_opponent_win_pct(prev_games)
		home_win_pct, home_games = get_home_win_pct(prev_games, home)

		stats += [win_pct] # season win pct
		stats += [home_win_pct] # home/away win pct

		stats += win_loss_convert(prev_games['WL'].tolist()) # last games win pct
		stats += point_diff # last games point diff
		stats += [pct for pct in opp_win_pct] # opponent win pct
		stats += opp_efg # opponent effective field goal percentage
		stats += prev_games['EFG'].tolist() # team effective field goal percentage



	# normal fully connected approach (with alternations)
	if NORMAL:
		#stats += [win_pct_convert(win_pct, len(prev_games))]
		stats += [win_pct]
		stats += prev_games['FTM'].tolist() # free throw makes
		stats += prev_games['FTA'].tolist() # free throw attempts
		stats += prev_games['OREB'].tolist() # offensive rebounds
		stats += prev_games['DREB'].tolist() # defensive rebounds
		stats += prev_games['TOV'].tolist() # turnovers
		stats += prev_games['EFG'].tolist()	# effective field goal percentage
		stats += prev_games['STL'].tolist() # steals
		stats += prev_games['BLK'].tolist() # blocks
		#stats += labels
		#stats += prev_games['AST'].tolist()

	return stats

def get_game_stats(id): 
	"""
	Accepts:
	game_id- the id of the game

	Returns:
	Array of needed stats for that game's vector
	"""

	game = GAME_TABLE[GAME_TABLE['Game_ID'] == id]

	#checks if the first row contains the away team
	if '@' in game.iloc[0]['MATCHUP']:
		away = get_team_stats(game.iloc[0], False)
		home = get_team_stats(game.iloc[1], True)

	else:
		away = get_team_stats(game.iloc[1], False)
		home = get_team_stats(game.iloc[0], True)

	if away == None or home == None:
		return None

	return home + away


def normalize(df):
	'''
	Accepts:
	df- dataframe 

	Returns:
	the dataframe normalized using min-max normalization
	'''

	result = df.copy()
	f = open("maxes.txt", "w")

	maxes = []
	mins = []
	for feature_name in df.columns:
	    max_value = df[feature_name].max()
	    min_value = df[feature_name].min()
	    result[feature_name] = (df[feature_name] - min_value) / (max_value - min_value)
	    maxes += [max_value]
	    mins += [min_value]

	f.write(', '.join(str(x) for x in maxes))
	f.write('\n\n')
	f.write(', '.join(str(x) for x in mins))

	return result

def add_weight(df):
	i = 0
	for column in df:
		if i > 1 and i < GAMES_BACK * NUM_FEATURES * 2 + 2:
			index = (GAMES_BACK - ((i - 2) % GAMES_BACK))
			df[column] = df[column] * (1 + (index/(GAMES_BACK**2)))
		i += 1
	return df


def save_input():
	'''
	Creates and saves a set of input vectors to use for training a network. Uses the global parameters defined above.
	'''

	stats = []

	games = set(GAME_TABLE['Game_ID'])
	i = 0

	#goes over each game
	for game_id in games:
		if i % 100 == 0:
			print(i)
		game_stats = get_game_stats(game_id)
		if game_stats == None:
			continue
		stats.append(game_stats + [LABELS[game_id]])
		i += 1


		
	df = pd.DataFrame(stats)
	#normalizes the data
	df = normalize(df)

	print("nans: ", df.isna().sum())
	#replaces nans with 0
	df = df.fillna(0)

	df = df.sample(frac = 1)

	if WEIGHTED:
		print(df)
		print(add_weight(df))
		df = add_weight(df)

	test_len = int(df.shape[0] * TEST_SPLIT)

	test_set = df.head(test_len)

	training_set = df.tail(df.shape[0] - test_len)

	suf = 'i'

	if CONVOLUTION:
		suf += '_conv' 

	if WIN_PCT:
		suf += '_win_pct'

	if WEIGHTED:
		suf += '_weighted'


	test_set.to_csv('team_input_tables/team_test_' + str(GAMES_BACK) + '_' + str(NUM_FEATURES) + '_' + str(MAX_DIFF) + '_' + str(FIRST_SEASON) + suf + '.csv', sep=',', index = False, header = None)
	training_set.to_csv('team_input_tables/team_training_' + str(GAMES_BACK) + '_' + str(NUM_FEATURES) + '_' + str(MAX_DIFF) + '_' + str(FIRST_SEASON) + suf + '.csv', sep=',', index = False, header = None)
	

#get dictionary of labels
LABELS = get_labels() 

def main():
	#init_team_games()
	#clean_team_games()
	save_input()

if __name__ == '__main__':
	main()