from nba_api.stats.endpoints import LeagueGameLog
from time import sleep
import pandas as pd
from ..config import SEASONS

def fetch_game_data():
    """Fetch NBA game data for specified seasons"""
    all_games = []
    
    for season in SEASONS:
        print(f"Fetching data for season {season}...")
        try:
            game_log = LeagueGameLog(season=season, timeout=30)
            data = game_log.get_data_frames()[0]
            all_games.append(data)
            sleep(2)  # Rate limiting
        except Exception as e:
            print(f"Error fetching season {season}: {e}")
            continue
    
    return pd.concat(all_games, ignore_index=True)