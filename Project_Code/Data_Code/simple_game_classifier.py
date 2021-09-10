import pandas as pd

GAMES = pd.read_csv('online_tables/games.csv', sep=',')

def check_win_pct(date, season, team):
	relevant_games = GAMES[GAMES['SEASON'] == season]
	relevant_games = relevant_games[relevant_games['GAME_DATE_EST'] < date]
	home_games = relevant_games[relevant_games['HOME_TEAM_ID'] == team]
	visitor_games =  relevant_games[relevant_games['VISITOR_TEAM_ID'] == team]

	hlen = len(home_games.index)
	vlen = len(visitor_games.index)
	if hlen == 0 or vlen == 0:
		return 0.5

	home_pct = 0
	for ind in home_games.index:
		if home_games['HOME_TEAM_WINS'][ind] == 1:
			home_pct += 1

	home_pct /= hlen

		
	visitor_pct = 0
	for ind in visitor_games.index:
		if visitor_games['HOME_TEAM_WINS'][ind] == 0:
			visitor_pct += 1

	visitor_pct /= vlen

	return (hlen * home_pct + vlen * visitor_pct)/(hlen + vlen)


def main():
	correct = 0
	for ind in GAMES.index:
		if ind%100 == 0: print(ind)
		home_pct = check_win_pct(GAMES['GAME_DATE_EST'][ind], GAMES['SEASON'][ind], GAMES['HOME_TEAM_ID'][ind])
		visitor_pct = check_win_pct(GAMES['GAME_DATE_EST'][ind], GAMES['SEASON'][ind], GAMES['VISITOR_TEAM_ID'][ind])

		if home_pct >= visitor_pct:
			guess = 1

		else:
			guess = 0

		if guess == GAMES['HOME_TEAM_WINS'][ind]:
			correct += 1


	print(correct/len(GAMES.index))


if __name__ == "__main__":
	main()

