# Importing modules and libraries
import pandas as pd
import numpy as np

# Opening raw dataset
pd.set_option('display.max_columns', None) # changes the maximum number of columns diplayed)
# Opending data
raw_nfl_data = pd.read_csv('resources/NFL_Play_by_Play_2009_2018.csv', low_memory=False)
raw_nfl_data.head()

# Prepping data for play-by-play analysis for next play prediction
play_by_play_data = raw_nfl_data[['game_id', 'game_date', 'home_team', 'yardline_100', 'qtr', 'half_seconds_remaining', 'game_seconds_remaining', 'down', 'ydstogo', 'posteam_timeouts_remaining', 'defteam_timeouts_remaining', 'score_differential', 'play_type']]
play_by_play_data.head()

#Opening zip code dataset
zip_code = pd.read_excel('resources/zip_team.xlsx',dtype = {"ZIP": str})
zip_code.head()

# Mapping zip codes to abbreviations
zip_code_mapping = zip_code.rename(columns={"ZIP": "zip_code", "Team Abbreviation": "home_team"})
zip_code_mapping.head()

# Merging play-by-play data with zip codes
play_by_play_data_w_zips = pd.merge(play_by_play_data, zip_code_mapping, on='home_team', how='left')
play_by_play_data_w_zips.head()

# Opening weather dataset
weather_data = pd.read_csv('resources/weather_final_full.csv')
weather_data.head()
weather_data = weather_data.drop(columns={'Unnamed: 0'})
weather_data = weather_data.rename(columns={"date": "game_date"})
weather_data.head()

# Converting zip code column to str for merging
weather_data['zip_code']=weather_data['zip_code'].astype(str)

# Merging nfl and weather tables
play_by_play_data_w_weather = pd.merge(play_by_play_data_w_zips, weather_data, on=['game_date','zip_code'], how='left')

play_by_play_data_w_weather.head()

# Removing all null values
final_play_by_play_data_w_weather = play_by_play_data_w_weather.dropna()
final_play_by_play_data_w_weather.head()

final_play_by_play_data_w_weather.isna().sum()

# Removing columns used for merging
reduced_final_play_by_play_data_w_weather = final_play_by_play_data_w_weather.drop(columns=['game_id', 'game_date', 'home_team', 'zip_code'], axis=1)

reduced_final_play_by_play_data_w_weather.columns

# Saving final table as csv
reduced_final_play_by_play_data_w_weather.to_csv('resources/final_table_for_model_build.csv')
