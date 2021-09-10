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

FIRST_SEASON = 2008

TEAM_TABLE = teams.get_teams()


def init_team_games():
	global GAME_TABLE
	teams = []
	for team in TEAM_TABLE:
		teams += [team['id']]

	df = teamgamelog.TeamGameLog(team_id = teams[0], season = SeasonAll.all).get_data_frames()[0]
	print(df)

	for team_id in teams[1:]:
		df = pd.concat([df, teamgamelog.TeamGameLog(team_id = team_id, season = SeasonAll.all).get_data_frames()[0]])
		print(team_id)

	df.to_csv('data_tables/team_data.csv', sep=',')


def clean_team_games():
	team_data = pd.read_csv('data_tables/team_data.csv', sep = ',')
	team_data['DATE_CONV'] = team_data['GAME_DATE'].apply(lambda x: (10000*int(x.split()[2]) + 100*month_conv(x.split()[0]) + int(x.split()[1][:2])))
	team_data = team_data[team_data['DATE_CONV'] > FIRST_SEASON*10000]
	team_data['EFG'] = (team_data["FGM"] + (1.5*team_data['FG3M']))/(team_data['FGA'] + team_data['FG3A'])
	team_data = team_data.drop(['MIN', 'FGM', 'FGA', 'FG_PCT', 'FG3M', 'FG3A', 'FG3_PCT', 'FT_PCT', 'REB', 'PF', 'W', 'L'], axis = 1)
	team_data = team_data.sort_values(by=['DATE_CONV'])
	print(team_data)
	team_data.to_csv('data_tables/team_data_clean_' + str(FIRST_SEASON) + '.csv', sep=',')



def month_conv(month):
	months = ["JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"]
	for i in range(len(months)):
		if months[i] == month:
			return i+1


	
def main():
	init_team_games()
	clean_team_games()


if __name__ == '__main__':
	main()