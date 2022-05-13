import glob
import os
from datetime import datetime
import pandas as pd
from bigquery import get_client
import configparser
import numpy as np


def filter_players_stats(player_stats):
    return player_stats.loc[player_stats['tInteractionDurationSec'] > 2]


def get_system_id():
    config_ini_path = "C:\Installation-Wizard\Installer\Config.ini"
    config_ini = configparser.ConfigParser()
    config_ini.optionxform = str
    if os.path.isfile(config_ini_path):
        config_ini.read(config_ini_path)
    return config_ini["Device"]["id"]


class DataSender:
    def __init__(self):
        self.json_key = 'analytics-fad3d92a636a.json'
        self.client = get_client(json_key_file=self.json_key, readonly=False, swallow_results=False)

    def insert_into_db(self, rows):
        # example: rows = [ {'game_type': 'hi', 'game_name': '1'} ]
        ret = self.client.push_rows('analytics', 'game_goodness_metrics', rows)
        print("num_rows", len(rows))
        print(ret)


def strided_app(a, L, S):  # Window len = L, Stride len/stepsize = S
    nrows = ((a.size - L) // S) + 1
    n = a.strides[0]
    return np.lib.stride_tricks.as_strided(a, shape=(nrows, L), strides=(S * n, n))


def calc_num_left_joined(players_stats, game_start, game_end):
    players_per_second = np.array([((second >= players_stats['tTSStart']) * \
                                    (second <= players_stats['tTSEnd'])).sum()
                                   for second in range(game_start, game_end + 1)])
    players_per_second_filtered = np.median(strided_app(players_per_second, 3, 1), axis=1)
    diff = np.diff(players_per_second_filtered, 1)
    return np.abs(diff[diff < 0].sum()), diff[diff > 0].sum()  # num left , num joined


if __name__ == '__main__':

    for system_id_int in [250487703, 316677320,
                          278190151, 344149346, 348020834, 368732374, 410463138,
                          513419919, 541686330, 563155083, 781435881]:  # get_system_id()
        # try:

        system_id = str(system_id_int)
        print("system id: ", system_id)
        with open(r"C:\Users\dan\Documents\projects\DanalyticsResults\\" + str(system_id) + "\\appSwitches.log",
                  mode="r") as f:
            switches_log = f.read()
        switches_log = switches_log.replace("start_time,end_time,last_app_name,current_app_name\n", "")
        with open(r"C:\Users\dan\Documents\projects\DanalyticsResults\\" + system_id + "\\appSwitches.log",
                  mode="w") as f:
            f.write(switches_log)

        game_switches = pd.read_csv(
            r"C:\Users\dan\Documents\projects\DanalyticsResults\\" + system_id + "\\appSwitches.log")
        game_switches.columns = ["start_time", "end_time", "last_app_type", "last_app_name",
                                 "current_app_type", "current_app_name"]
        for stats_log in glob.glob(r"C:\Users\dan\Documents\projects\DanalyticsResults\\" + system_id + "\\*2018.log"):

            players_stats = pd.read_csv(stats_log)
            players_stats.columns = ['haspId', 'szCurrApplicationType', 'szCurrApplicationName',
                                     'uiInteractionID', 'uiHumanID', 'tTSStart', 'tTSEnd', 'tInteractionDurationSec',
                                     'ucHumanHeight',
                                     'uiTrackDistanceMeters', 'ucTrackCoveragePercentage']

            # players_stats = filter_players_stats(players_stats)
            stats = []
            for _, row in game_switches.iterrows():
                stats_dictionary = {}
                players_stats_during_game = players_stats.loc[(players_stats['tTSEnd'] <= row['end_time']) *
                                                              (players_stats['tTSStart'] >= row['start_time'])]

                if len(players_stats_during_game) > 0:
                    num_players_left, num_players_joined = calc_num_left_joined(players_stats_during_game,
                                                                                row['start_time'], row['end_time'])
                    stats_dictionary["game_type"] = row['current_app_type']
                    stats_dictionary["game_name"] = row['current_app_name']
                    stats_dictionary["system_id"] = system_id
                    stats_dictionary["game_start"] = datetime.utcfromtimestamp(row['start_time']).strftime(
                        '%Y-%m-%dT%H:%M:%S')
                    stats_dictionary["game_end"] = datetime.utcfromtimestamp(row['end_time']).strftime(
                        '%Y-%m-%dT%H:%M:%S')
                    stats_dictionary["num_players_joined"] = int(num_players_joined)
                    stats_dictionary["num_players_left"] = int(num_players_left)
                    stats_dictionary["game_duration_seconds"] = int(row['end_time'] - row['start_time'])
                    stats_dictionary["mean_speed_m_per_s"] = float((players_stats_during_game['uiTrackDistanceMeters'] / \
                                                                    (players_stats_during_game['tTSEnd'] -
                                                                     players_stats_during_game['tTSStart'])).mean())
                    if not np.isnan(stats_dictionary["mean_speed_m_per_s"]) and\
                            not np.isinf(stats_dictionary["mean_speed_m_per_s"]):
                        stats.append(stats_dictionary)
            data_sender = DataSender()
            data_sender.insert_into_db(stats)
        # except:
        #     print(system_id_int)
