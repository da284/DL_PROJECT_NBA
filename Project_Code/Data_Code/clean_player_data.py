import pandas as pd

GAMES = pd.read_csv('online_tables/games.csv', sep=',')
GAMES_DETAILS = pd.read_csv('online_tables/games_details.csv', sep=',')
GAMES_DETAILS = pd.merge(pd.DataFrame(GAMES, columns = ['GAME_DATE_EST', 'GAME_ID']), GAMES_DETAILS, on = 'GAME_ID')
GAMES_DETAILS = GAMES_DETAILS.fillna('0')
GAMES_DETAILS = GAMES_DETAILS.sort_values(['GAME_DATE_EST']) 


def get_secs(minutes):
	'''
	converts player minutes to seconds (more comfortable format)
	'''
	times = minutes.split(':')
	if len(times) == 1:
		return int(times[0]) * 60
	return (int(times[0]) * 60 + int(times[1]))/1000


def main():
	series = GAMES_DETAILS['MIN']
	col = [0 for i in range(len(series))]

	for i in range(len(series)):
		if i%10000==0:
			print(i)

		col[i] = get_secs(series[i])

	GAMES_DETAILS = GAMES_DETAILS.drop(columns = ['MIN']) # remove minutes column
	GAMES_DETAILS = GAMES_DETAILS.assign(SECS = col) # adds seconds column

	GAMES_DETAILS.to_csv('online_tables/clean_game_details.csv', sep=',')


if __name__ == '__main__':
	main()

