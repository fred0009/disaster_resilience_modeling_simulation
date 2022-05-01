import pickle
import os
from main import Results
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np

def visualize_results(data1, data2, data3, sim_t):
    fig = plt.figure(figsize=(8, 6), dpi=100)
    ax_untreated = fig.add_subplot(311)
    ax_treated = fig.add_subplot(312)
    ax_critical_functionality = fig.add_subplot(313)
    t = list(range(round(sim_t/30)))


    ax_untreated.plot( t, np.mean(data1, axis=0), 'r')
    ax_treated.plot( t, np.mean(data2, axis=0), 'g')
    ax_critical_functionality.plot( t, np.mean(data3, axis=0), 'blue')

    plt.show(fig)
    plt.close()

# open a file, where you stored the pickled data
cwd = os.getcwd()
cwd = cwd[:-3]
os.chdir(cwd)
comduration = 45
victims = 100
scenario = 'DHO'
strategy = 'nearest'
disn = 4

filename1 = 'results/uv/ComDuration_{}__Victims_{}__Scenario_{}__Strategy_{}__DisSiteN__{}.npy'.format(comduration,
                                                                 victims,scenario, strategy, disn)
filename2 = 'results/tv/ComDuration_{}__Victims_{}__Scenario_{}__Strategy_{}__DisSiteN__{}.npy'.format(comduration,
                                                                 victims,scenario, strategy, disn)
filename3 = 'results/cf/ComDuration_{}__Victims_{}__Scenario_{}__Strategy_{}__DisSiteN__{}.npy'.format(comduration,
                                                                 victims,scenario, strategy, disn)
data1 = np.load(filename1)
data2 = np.load(filename2)
data3 = np.load(filename3)

SIMULATION_DURATION = 60 * 120

visualize_results(data1, data2, data3, SIMULATION_DURATION)
