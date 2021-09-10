import pandas as pd
import numpy as np
import time

PLAYERS_PER_TEAM = 3
GAMES_BACK = 8
MAX_DIFF = 25
VEC_LEN = 12
TEST_SPLIT = 0.1

GAMES = pd.read_csv('online_tables/games.csv', sep=',')
GAMES_DETAILS = pd.read_csv('online_tables/clean_game_details.csv', sep=',') 
PLAYER_DICT = dict()


def calc_label(home_pts, away_pts):
	return max(min(1.0, (home_pts - away_pts + MAX_DIFF)/(2 * MAX_DIFF)),0.0)

def get_labels():
	df = GAMES
	d = {}
	for i in range(len(df)):
		d[df.iloc[i]['GAME_ID']] = calc_label(df.iloc[i]['PTS_home'], df.iloc[i]['PTS_away'])

	return d


def get_roster(table, team_id): #returns the PLAYERS_PER_TEAM ids of the players that played the most minutes

	table = table.loc[table['TEAM_ID'] == team_id].sort_values('SECS', ascending = False)
	return table.head(PLAYERS_PER_TEAM).PLAYER_ID.tolist()

def get_player_stats(player_id, date): #returns stat vector for a specific player
		if player_id not in PLAYER_DICT:
			PLAYER_DICT[player_id] = GAMES_DETAILS.loc[GAMES_DETAILS['PLAYER_ID'] == player_id].reset_index(drop=True)

		player_games = PLAYER_DICT[player_id]	
		game_index = player_games.index[player_games['GAME_DATE_EST'] == date].tolist()[0]
		#player_games = PLAYER_DICT[player_id].loc[GAMES_DETAILS['GAME_DATE_EST'] < date]

		if game_index == 0:
			return [0] * VEC_LEN

		player_games = player_games[game_index - min(GAMES_BACK, game_index): game_index]
		averages = player_games.mean() #takes the relevant columns of the averages of these games
		#print("averges: ", averages)
		return [averages.get(key=p) for p in ['FGM', 'FGA', 'FG3M', 'FG3A', 'FTM', 'FTA', 'OREB', 'DREB', 'AST', 'STL', 'BLK', 'TO']]

def get_game_stats(game_id, label): #returns stat vector for a specific game 
		game_table = GAMES_DETAILS.loc[GAMES_DETAILS['GAME_ID'] == game_id] #stats for each player in the game
		game = GAMES.loc[GAMES['GAME_ID'] == game_id] #the game (general)
		home_team = game.HOME_TEAM_ID.values[0] #home team
		vis_team = game.VISITOR_TEAM_ID.values[0] #visiting team
		roster = get_roster(game_table, home_team) + get_roster(game_table, vis_team) #creates a vector with all relevant players
		input_vector = []
		
		for player in roster:
			input_vector += get_player_stats(player, game.GAME_DATE_EST.values[0])
		

		return input_vector + [label]


def normalize(df):
    result = df.copy()
    for feature_name in df.columns:
        max_value = df[feature_name].max()
        min_value = df[feature_name].min()
        result[feature_name] = (df[feature_name] - min_value) / (max_value - min_value)



    return result

def main():
	game_ids = GAMES.GAME_ID.unique()
	labels = get_labels()
	inputs = [get_game_stats(game_ids[0], labels[game_ids[0]])]

	for i in range(len(game_ids[1:])):
		if i % 100 == 0:
			print(i)
		inputs.append(get_game_stats(game_ids[i], labels[game_ids[i]]))


	inputs = pd.DataFrame(inputs)
	inputs = normalize(inputs)
	inputs = inputs.fillna(0)

	inputs = inputs.sample(frac = 1)

	test_len = int(inputs.shape[0] * TEST_SPLIT)

	test_set = inputs.head(test_len)

	training_set = inputs.tail(inputs.shape[0] - test_len)

	test_set.to_csv('player_input_tables/player_test_' + str(PLAYERS_PER_TEAM) + '_' + str(GAMES_BACK) + '_' + str(MAX_DIFF) + '.csv', sep=',', header=False, index=False)
	training_set.to_csv('player_input_tables/player_training_' + str(PLAYERS_PER_TEAM) + '_' + str(GAMES_BACK) + '_' + str(MAX_DIFF) + '.csv', sep=',', header=False, index=False)


if __name__ == '__main__':
	main()