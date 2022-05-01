import numpy as np
import copy
import random
import pandas as pd
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import os
from operator import itemgetter
import run_simulation as run_sim
from scipy.stats import truncnorm
import simpy
import pickle
from multiprocessing import Process

class Results:
    def __init__(self, total_simulation_duration, n_sim):
        self.n_sim = n_sim
        step = 30
        simulation_duration = round(total_simulation_duration/step)
        self.results_untreated_victims = np.zeros(simulation_duration)
        self.results_treated_victims = np.zeros(simulation_duration)
        self.results_critical_functionality = np.ones(simulation_duration)

    def append_values(self, sim_result):
        self.results_untreated_victims = np.vstack( (self.results_untreated_victims, np.array(sim_result.total_untreated_victims_series)) )
        self.results_treated_victims = np.vstack( (self.results_treated_victims, np.array(sim_result.total_treated_victims_series)) )
        self.results_critical_functionality = np.vstack( (self.results_critical_functionality, np.array(sim_result.critical_functionality_series)) )

    def clean_initial_values(self):
        self.results_untreated_victims = np.delete(self.results_untreated_victims, (0), axis=0)
        self.results_treated_victims = np.delete(self.results_treated_victims, (0), axis=0)
        self.results_critical_functionality = np.delete(self.results_critical_functionality, (0), axis=0)

    def save_object(self, com_duration, victims, scenario, strategy, n_disaster_site):
        filename1 = 'results/uv/ComDuration_{}__Victims_{}__Scenario_{}__Strategy_{}__DisSiteN__{}.npy'.format(com_duration,
                                                                 victims,scenario, strategy, n_disaster_site)
        filename2 = 'results/tv/ComDuration_{}__Victims_{}__Scenario_{}__Strategy_{}__DisSiteN__{}.npy'.format(
            com_duration,
            victims, scenario, strategy, n_disaster_site)
        filename3 = 'results/cf/ComDuration_{}__Victims_{}__Scenario_{}__Strategy_{}__DisSiteN__{}.npy'.format(
            com_duration,
            victims, scenario, strategy, n_disaster_site)
        np.save(filename1, self.results_untreated_victims)
        np.save(filename2, self.results_treated_victims)
        np.save(filename3, self.results_critical_functionality)

class Hospital:
    def __init__(self, hospital_name, city, beds):
        self.ID = hospital_name
        self.INITIAL_BEDS = beds
        self.available_beds = beds
        self.city = city
        self.health_office = None
        self.treated_victims = 0
        self.untreated_victims = 0
        self.untreated_victims_on_the_way = 0
        self.activated_status = False
        self.neighbors = []
        self.disasters_and_distance = {}


class Disaster:
    def __init__(self, ID, victims):
        self.ID = ID
        self.victims = victims
        self.hospital_and_driving_time = {}


class PublicHealthOffice:
    def __init__(self, hospital_list, env):
        self.ID = 'PHO'
        self.hospitals_in_administration = hospital_list
        self.communication_channel = simpy.Resource(env, capacity=1)


class DistrictHealthOffice:
    def __init__(self, city, hospital_list, PHO, env):
        self.ID = city
        self.PHO = PHO
        self.hospitals_in_administration = [x for x in hospital_list if x.city==city]
        self.communication_channel = simpy.Resource(env, capacity=1)

        for hospital in self.hospitals_in_administration:
            hospital.health_office = self


def import_data():
    # Get driving time data
    cwd = os.getcwd()
    cwd = cwd[:-3]
    os.chdir(cwd)

    testing = False
    # testing = True

    if testing == False:
        df_disaster = pd.read_excel(r'data/raw/Disaster Location.xls')
        df_driving_time = pd.read_csv(r'data/processed/Driving_Time_Matrix.csv')
        df = pd.read_excel(r'data/processed/Simplified_Hospital_Data.xls')
    else:
        df_disaster = pd.read_excel('data/testing_simulation/Disaster Location Testing.xls', index=False)
        df_driving_time = pd.read_csv('data/testing_simulation/Driving_Time_Matrix Testing.csv')
        df = pd.read_excel('data/testing_simulation/Simplified_Hospital_Data Testing.xls', index=False)

    available_beds_col = 'AVAILABLE BEDS'

    return df, df_driving_time, df_disaster, available_beds_col


def generate_hospitals(df, available_beds_col):
    hospital_objects = [ Hospital( row['NAMA RS'], row['KAB/KOTA'], int(row[available_beds_col])) for
                         index, row in df.iterrows()]
    return hospital_objects


def set_neighbors_standard(df_driving_time, TIME_OF_THE_DAY, hospital_list):
    central_hospitals_ids = ['RSUP Fatmawati', 'RSUP Persahabatan', 'RSUPN Dr. Cipto Mangunkusumo']
    central_hospitals = [x for x in hospital_list if x.ID in central_hospitals_ids]

    for hospital in hospital_list:
        nearest_nat_gen_hosp = []
        for priority_hospital in central_hospitals:
            if hospital != priority_hospital:
                od_pair = hospital.ID + ',' + priority_hospital.ID
                driving_time = int(df_driving_time[df_driving_time['OD Pair'] == od_pair][TIME_OF_THE_DAY].values)
                nearest_nat_gen_hosp.append( (priority_hospital,driving_time) )
        hospital.neighbors = sorted( nearest_nat_gen_hosp, key=itemgetter(1) )

        neighbor_driving_time = []
        for other_hospital in hospital_list:
            if hospital != other_hospital and other_hospital not in central_hospitals:
                od_pair = hospital.ID + ',' + other_hospital.ID
                driving_time = int(df_driving_time[df_driving_time['OD Pair'] == od_pair][TIME_OF_THE_DAY].values)
                neighbor_driving_time.append( (other_hospital,driving_time) )
        hospital.neighbors += sorted( neighbor_driving_time, key=itemgetter(1) )
    return central_hospitals_ids

def set_neighbors_nearest(df_driving_time, TIME_OF_THE_DAY, hospital_list):
    for hospital in hospital_list:
        neighbor_driving_time = []
        for other_hospital in hospital_list:
            if hospital != other_hospital:
                od_pair = hospital.ID + ',' + other_hospital.ID
                driving_time = int(df_driving_time[df_driving_time['OD Pair'] == od_pair][TIME_OF_THE_DAY].values)
                neighbor_driving_time.append( (other_hospital,driving_time) )
        hospital.neighbors = sorted( neighbor_driving_time, key=itemgetter(1) )
    return []


def set_responding_hospitals(victims, disaster_list, hospital_list, df_disaster, df_driving_time, TIME_OF_THE_DAY,
                             TOLERANCE_TIME_VICINITY):
    responding_hospitals = []
    disasters = []
    for disloc in disaster_list:
        disaster_object = Disaster(disloc, victims)
        min_drive_time = int(df_driving_time[TIME_OF_THE_DAY][df_driving_time['OD Pair'].str.startswith(disloc) ].min())
        for hospital in hospital_list:
            od_pair = disloc+','+ hospital.ID
            od_driving_time = int(df_driving_time[TIME_OF_THE_DAY][df_driving_time['OD Pair']==od_pair])
            if od_driving_time < min_drive_time + TOLERANCE_TIME_VICINITY :
                disaster_object.hospital_and_driving_time.update( {hospital: od_driving_time} )
                responding_hospitals.append(hospital)
                hospital.activated_status = True
        disasters.append( disaster_object )

    return list(set(responding_hospitals)), disasters


def visualize_results(results, sim_t):
    print(results.results_untreated_victims)
    fig = plt.figure(figsize=(8, 6), dpi=100)
    ax_untreated = fig.add_subplot(311)
    ax_treated = fig.add_subplot(312)
    ax_critical_functionality = fig.add_subplot(313)
    t = list(range(sim_t))

    ax_untreated.plot( t, np.mean(results.results_untreated_victims, axis=0), 'r')
    ax_treated.plot( t, np.mean(results.results_treated_victims, axis=0), 'g')
    ax_critical_functionality.plot( t, np.mean(results.results_critical_functionality, axis=0), 'blue')

    plt.show(fig)
    plt.close()


def main(df, df_disaster, df_driving_time, beds_col, **kwargs):
    TIME_OF_THE_DAY = '12'         # 6 pm. 0 starts at 6am
    TOLERANCE_TIME_VICINITY = 300  # 5 mins
    SIMULATION_DURATION = 60 * 120  # 2 hours simulations
    N_SIM = 1000

    # Get all parameters
    params = {}
    for key,val in kwargs.items():
        params[key] = val
    victims = params['victims']
    scenario = params['scenario']
    com_duration = params['com_duration']
    strategy = params['strategy']
    n_disaster_site = params['disaster_site']


    # print('Generate Actors...')
    hospital_list_original = generate_hospitals(df, beds_col)
    results_object = Results(SIMULATION_DURATION, N_SIM)

    if scenario == 'PHO':
        # print('Set hospital neighbors...')
        if strategy == 'Standard':
            central_hospitals = set_neighbors_standard(df_driving_time, TIME_OF_THE_DAY, hospital_list_original)
        elif strategy == 'Nearest':
            central_hospitals = set_neighbors_nearest(df_driving_time, TIME_OF_THE_DAY, hospital_list_original)

        # print('Start Simulation...')
        for i in range(N_SIM):
            # print('Simulation {}'.format(i))
            env = simpy.Environment()
            hospital_list = copy.deepcopy(hospital_list_original)
            disaster_locations = random.sample( list(df_disaster['Location Name']), n_disaster_site)
            responding_hospitals, disasters = set_responding_hospitals(victims, disaster_locations, hospital_list,
                                                                       df_disaster, df_driving_time, TIME_OF_THE_DAY,
                                                                        TOLERANCE_TIME_VICINITY)
            PHO = PublicHealthOffice(hospital_list, env)


            sim_result = run_sim.start_simulation(disasters, responding_hospitals, hospital_list, PHO,
                                                  com_duration, central_hospitals, scenario, env)
            env.run(until=SIMULATION_DURATION)

            results_object.append_values(sim_result)


    elif scenario == 'DHO': # Scenario DHO means DHO + PHO with nearest hospital setting (not Central Hospital strategy)
        # print('Set hospital neighbors...')
        central_hospitals = set_neighbors_nearest(df_driving_time, TIME_OF_THE_DAY, hospital_list_original)

        # print('Start Simulation...')
        for i in range(N_SIM):
            # print('Simulation {}'.format(i))
            env = simpy.Environment()
            hospital_list = copy.deepcopy(hospital_list_original)
            disaster_locations = random.sample(list(df_disaster['Location Name']), 7)
            responding_hospitals, disasters = set_responding_hospitals(victims, disaster_locations, hospital_list,
                                                                       df_disaster, df_driving_time, TIME_OF_THE_DAY,
                                                                       TOLERANCE_TIME_VICINITY)

            PHO = PublicHealthOffice(hospital_list, env)
            cities = list(df['KAB/KOTA'].unique())
            DHOs = {}
            for city in cities:
                DHO = DistrictHealthOffice(city, hospital_list, PHO, env)
                DHOs[city] = DHO


            sim_result = run_sim.start_simulation(disasters, responding_hospitals, hospital_list, DHOs,
                                                  com_duration, central_hospitals, scenario, env)
            env.run(until=SIMULATION_DURATION)

            results_object.append_values(sim_result)

    results_object.clean_initial_values()
    results_object.save_object(com_duration, victims, scenario, strategy, n_disaster_site)
    # f = open('results/ComDuration_{}__Victims_{}__Scenario_{}__Strategy_{}__DisSiteN__{}.pkl'.format(com_duration, victims,
    #                                                                                scenario, strategy, n_disaster_site), 'wb')
    # pickle.dump(results_object, f, pickle.HIGHEST_PROTOCOL)
    # f.close()

    # visualize_results(results_object, SIMULATION_DURATION)


if __name__ == '__main__':

    df, df_driving_time, df_disaster, available_beds_col = import_data()

    # Max Victim 7733
    victim_list = [100, 250, 500, 750, 1000, 1250, 1500]
    disaster_site_list = [1,2,3,4,5,6,7]
    com_duration_list = [5, 15, 30, 45, 60, 75, 90]
    scenario_list = ['PHO', 'DHO']   # Scenario DHO means DHO + PHO with nearest hospital setting (not Central Hospital strategy)

    # victim_list = [750]
    # disaster_site_list = [4]
    # com_duration_list = [45]
    # scenario_list = ['DHO']

    # scenario = 'DHO'
    # strategy = 'Standard'
    # strategy_list = [None]

    for victim in victim_list:
        for com_duration in com_duration_list:
            procs = []
            for scenario in scenario_list:
                if scenario == 'PHO':
                    strategy_list = ['Standard', 'Nearest']
                elif scenario == 'DHO':   # Scenario DHO means DHO + PHO with nearest hospital setting (not Central Hospital strategy)
                    strategy_list = ['Nearest']

                for strategy in strategy_list:
                    for disaster_site in disaster_site_list:
                        print(scenario, strategy, victim, com_duration, disaster_site)
                        # main(df, df_disaster, df_driving_time, available_beds_col, victims=victim, scenario=scenario,
                        #     com_duration=com_duration, strategy=strategy, disaster_site=disaster_site)
                        params = {'victims':victim, "scenario":scenario, "com_duration":com_duration, "strategy":strategy,
                                                                                             "disaster_site":disaster_site}
                        proc = Process(target=main, args=[df, df_disaster, df_driving_time, available_beds_col],
                                                          kwargs=params)
                        procs.append(proc)
                        proc.start()
            print(len(procs))
            for proc in procs:
                proc.join()