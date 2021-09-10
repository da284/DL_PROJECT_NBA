import sys
import subprocess

# implement pip as a subprocess:
subprocess.check_call([sys.executable, '-m', 'pip', 'install', 
'pysimplegui'])
subprocess.check_call([sys.executable, '-m', 'pip', 'install', 
'nba_api'])
subprocess.check_call([sys.executable, '-m', 'pip', 'install', 
'tensorflow'])
subprocess.check_call([sys.executable, '-m', 'pip', 'install', 
'numpy'])
subprocess.check_call([sys.executable, '-m', 'pip', 'install', 
'numpy'])
subprocess.check_call([sys.executable, '-m', 'pip', 'install', 
'pandas'])

import PySimpleGUI as sg
import numpy as np
import pandas
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense
from nba_api.stats.static import teams
from nba_api.stats.endpoints import teamgamelog
from nba_api.stats.library.parameters import SeasonAll


THRESHOLD_ARRAY = [(0.08,58.8),(0.12,65.8),(0.5,75.8)]
DENSE1 = 75
DENSE2 = 75
DENSE3 = 75
DENSE4 = 75
ABBREVIATION_LEN = 3
DATE_RANGE = [[1,31],[1,12],[2004,2021]]
FEATURES_LEN = 130
WINDOW_SIZE = (500, 300)


MAXES = [1.0, 52.0, 52.0, 52.0, 52.0, 48.0, 52.0, 50.0, 50.0, 64.0, 64.0, 64.0, 64.0, 59.0, 64.0, 63.0, 64.0, 38.0, 38.0, 38.0, 38.0, 28.0, 38.0, 30.0, 38.0, 55.0, 55.0, 55.0, 56.0, 56.0, 56.0, 56.0, 56.0, 30.0, 30.0, 30.0, 30.0, 30.0, 30.0, 30.0, 29.0, 0.7261904761904762, 0.7375, 0.7375, 0.7375, 0.7548076923076923, 0.7375, 0.7375, 0.7548076923076923, 22.0, 22.0, 21.0, 21.0, 22.0, 22.0, 22.0, 21.0, 20.0, 19.0, 19.0, 20.0, 20.0, 19.0, 19.0, 19.0, 1.0, 50.0, 48.0, 50.0, 48.0, 52.0, 50.0, 52.0, 52.0, 64.0, 63.0, 63.0, 63.0, 64.0, 64.0, 64.0, 64.0, 30.0, 31.0, 30.0, 31.0, 38.0, 31.0, 38.0, 31.0, 56.0, 56.0, 56.0, 55.0, 55.0, 55.0, 55.0, 55.0, 30.0, 29.0, 30.0, 30.0, 30.0, 30.0, 30.0, 30.0, 0.7548076923076923, 0.7548076923076923, 0.7548076923076923, 0.7548076923076923, 0.7375, 0.7548076923076923, 0.7548076923076923, 0.7300884955752213, 22.0, 22.0, 22.0, 22.0, 22.0, 21.0, 22.0, 22.0, 18.0, 20.0, 20.0, 19.0, 19.0, 20.0, 20.0, 20.0, 1.0]
MINS = [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 12.0, 0.0, 0.0, 12.0, 0.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 1.0, 0.2265625, 0.2265625, 0.2422680412371134, 0.2422680412371134, 0.2265625, 0.2265625, 0.2265625, 0.2265625, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 13.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.2336448598130841, 0.24609375, 0.2265625, 0.2265625, 0.2336448598130841, 0.2422680412371134, 0.2336448598130841, 0.2336448598130841, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]


def build_model():
	model = Sequential()
	model.add(Dense(DENSE1, input_dim=FEATURES_LEN, activation='relu'))
	model.add(Dense(DENSE2, activation='relu'))
	model.add(Dense(DENSE3, activation='relu'))
	model.add(Dense(DENSE4, activation='relu'))
	model.add(Dense(1, activation='sigmoid'))
	return model

def validate_input(values):
	#check if the input is valid
	if len(values[0]) != ABBREVIATION_LEN or len(values[1]) != ABBREVIATION_LEN:
		return False
	if not values[0].isupper() or not values[1].isupper():
		return False
	if values[0] == values[1]:
		return False
	if values[2].count("/") != 2:
		return False
	day, month, year = values[2].split("/")
	for element, date_range in zip([day,month,year], DATE_RANGE):
		if not element.isnumeric():
			return False
		if not date_range[0] <= int(element) <= date_range[1]:
			return False

	return True


def month_conv(month):
	months = ["JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"]
	for i in range(len(months)):
		if months[i] == month:
			return i+1


def get_team_stats(id, date):
	date_arr = [int(num) for num in date.split('/')]
	date_conv = 10000*date_arr[2] + 100*date_arr[1] + date_arr[0]
	team_games = teamgamelog.TeamGameLog(team_id = id, season = SeasonAll.all).get_data_frames()[0]
	team_games['DATE_CONV'] = team_games['GAME_DATE'].apply(lambda x: (10000*int(x.split()[2]) + 100*month_conv(x.split()[0]) + int(x.split()[1][:2])))

	prev_games = team_games[team_games['DATE_CONV'] < date_conv].head(8)
	prev_games['EFG'] = (prev_games["FGM"] + (1.5*prev_games['FG3M']))/(prev_games['FGA'] + prev_games['FG3A'])

	stats = []

	stats += [prev_games.iloc[0]['W_PCT']] # win percentage
	stats += prev_games['FTM'].tolist() # free throw makes
	stats += prev_games['FTA'].tolist() # free throw attempts
	stats += prev_games['OREB'].tolist() # offensive rebounds
	stats += prev_games['DREB'].tolist() # defensive rebounds
	stats += prev_games['TOV'].tolist() # turnovers
	stats += prev_games['EFG'].tolist()	# effective field goal percentage
	stats += prev_games['STL'].tolist() # steals
	stats += prev_games['BLK'].tolist() # blocks

	for i in range(len(stats)):
		stats[i] = (stats[i] - MINS[i])/(MAXES[i] - MINS[i])

	return stats


def get_data(values):
	# Parameters: values from the user, value[0] = team1, values[1] = team2, values[2] = date
	# Return: vector of data for this specific game
	home_team_abb = values[0]
	away_team_abb = values[1]
	date = values[2]

	team_dict = teams.get_teams()

	home_id = 0
	away_id = 0

	for team in team_dict:
		if home_team_abb.upper() == team['abbreviation'].upper():
			home_id = team['id']

		if away_team_abb.upper() == team['abbreviation'].upper():
			away_id = team['id']

	if home_id == 0 or away_id == 0:
		get_request(resubmit=True)


	home_stats = get_team_stats(home_id, date)
	away_stats = get_team_stats(away_id, date)

	return np.array(home_stats + away_stats).reshape((1,FEATURES_LEN))

def cancel():
	layout = [[sg.Text('Bye!')],          
	                 [sg.Exit()]]
	window = sg.Window('NBA Games Predictor', layout, element_justification='c')    
	event, values = window.read()
	window.close()
	quit()

def calc_pr(label):
	distance = abs(label-0.5)
	for threshold, pr in THRESHOLD_ARRAY:
		if distance <= threshold:
			return pr


def predict_game(values):
	vector = get_data(values)
	model = build_model()
	filepath = "model_weights.hdf5"
	model.load_weights(filepath)
	label = model.predict(vector)
	if label > 0.5:
		team = values[1]#away team winning
	else:
		team = values[0]#home team winning
	pr = calc_pr(label)
	show_result(team, pr)

def show_result(team, pr):
	layout = [[sg.Text('The winning team is: {0}\nWith probability: {1}'.format(team, pr))],
				[sg.Text('Press OK to go back')],          
	                 [sg.OK(), sg.Cancel()]]

	window = sg.Window('NBA Games Predictor', layout, size=WINDOW_SIZE, element_justification='c')    
	event, values = window.read()
	window.close()
	if event == 'Cancel':
		cancel()
	if event == 'OK':
		get_request()

def get_request(resubmit=False):
	layout = [[sg.Text('Enter two teams e.g. BOS, LAL')],
				[sg.Text('Home team:')],
	            [sg.InputText()],
				[sg.Text('Away team:')],    
	            [sg.InputText()],
	            [sg.Text('Enter date between 2004-2021, e.g. 25/04/2021')],      
	            [sg.InputText()],      
	            [sg.Submit(), sg.Cancel()]]

	if resubmit:
		layout.append([sg.Text('INVALID INPUT!')])
    
	window = sg.Window('NBA Games Predictor', layout, size=WINDOW_SIZE, element_justification='c')    
	event, values = window.read()
	window.close()

	if event == 'Cancel':
		cancel()
	if event == 'Submit':
		if not validate_input(values):
			get_request(resubmit=True)
		predict_game(values)

def main():
	get_request()

if __name__ == '__main__':
	main()
