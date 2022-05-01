import numpy as np
import pandas as pd
import folium
import branca
from folium import plugins
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
import geojsoncontour
import scipy as sp
import scipy.ndimage
import networkx as nx
from collections import namedtuple
import os
import json
import random
import src.run_simulation as run_sim

def set_neighbors(G):
    DRIVING_TIME_MAX_LIMIT = 1800
    TIME_OF_THE_DAY = '4'
    df_driving_time = pd.read_csv('data/processed/Driving_Time_Matrix.csv')
    for receiver in G.nodes():
        for node in G.nodes():
            if receiver != node:
                od_pair = G.node[node]['NAMA RS'] + ',' + G.node[receiver]['NAMA RS']
                driving_time = int(df_driving_time[df_driving_time['OD Pair'] == od_pair][TIME_OF_THE_DAY].values)
                if driving_time < DRIVING_TIME_MAX_LIMIT:
                    G.node[receiver]['neighbors'].append(node)


def generate_initial_links(MAX_EDGES, H):
    G = nx.Graph()
    G.add_nodes_from(H.nodes(data=True))
    print(G.nodes(data=True))
    for node in G:
        for i in range(100):
            try:
                node2 = random.choice(G.node[node]['neighbors'])
            except IndexError as e:
                print(e)
                break
            if len(G[node]) < MAX_EDGES:
                if len(G[node2]) < MAX_EDGES:
                    G.add_edge(node, node2)
            else:
                break
    return G

def get_bearing(p1, p2):
    '''
    Returns compass bearing from p1 to p2

    Parameters
    p1 : namedtuple with lat lon
    p2 : namedtuple with lat lon

    Return
    compass bearing of type float

    Notes
    Based on https://gist.github.com/jeromer/2005586
    '''

    long_diff = np.radians(p2.lon - p1.lon)

    lat1 = np.radians(p1.lat)
    lat2 = np.radians(p2.lat)

    x = np.sin(long_diff) * np.cos(lat2)
    y = (np.cos(lat1) * np.sin(lat2)
         - (np.sin(lat1) * np.cos(lat2)
            * np.cos(long_diff)))
    bearing = np.degrees(np.arctan2(x, y))

    # adjusting for compass bearing
    if bearing < 0:
        return bearing + 360
    return bearing

def get_arrows(some_map, locations, color='black', size=4, n_arrows=3):
    '''
    Get a list of correctly placed and rotated
    arrows/markers to be plotted

    Parameters
    locations : list of lists of lat lons that represent the
                start and end of the line.
                eg [[41.1132, -96.1993],[41.3810, -95.8021]]
    arrow_color : default is 'blue'
    size : default is 6
    n_arrows : number of arrows to create.  default is 3
    Return
    list of arrows/markers
    '''

    Point = namedtuple('Point', field_names=['lat', 'lon'])

    # creating point from our Point named tuple
    p1 = Point(locations[0][0], locations[0][1])
    p2 = Point(locations[1][0], locations[1][1])

    # getting the rotation needed for our marker.
    # Subtracting 90 to account for the marker's orientation
    # of due East(get_bearing returns North)
    rotation = get_bearing(p1, p2) - 90

    # get an evenly space list of lats and lons for our arrows
    # note that I'm discarding the first and last for aesthetics
    # as I'm using markers to denote the start and end
    arrow_lats = np.linspace(p1.lat, p2.lat, n_arrows + 2)[1:n_arrows + 1]
    arrow_lons = np.linspace(p1.lon, p2.lon, n_arrows + 2)[1:n_arrows + 1]

    arrows = []

    # creating each "arrow" and appending them to our arrows list
    for points in zip(arrow_lats, arrow_lons):
        arrows.append(folium.RegularPolygonMarker(location=points,
                                                  color=color, fill_color=color, opacity=0.5, number_of_sides=3,
                                                  fill_opacity=0.5, radius=size, rotation=rotation).add_to(some_map))
    return arrows



# Setting Working Directory
cwd = os.getcwd()
cwd = cwd[:-3]
os.chdir(cwd)

# Get Disaster Location
code = 22
df_dis = pd.read_excel('data/raw/Disaster Location.xls', index=False)
locationlist = df_dis[ ['Location Name','Latitude','Longitude', 'Group'] ]
locationlist = locationlist.values.tolist()
locationlist = [x for x in locationlist if x[3]==code]

# Load Network
# G_opt = nx.read_gpickle("results/stability/LONG_BESTOPTIMUM_2_2500.gpickle")
# G_opt = nx.read_gpickle("results/stability/server_best_optimum_2000_L2.gpickle")
G_opt = nx.read_gpickle("results/stability/server_best_optimum_model22_DisVic_300DriveLim_1800_FINAL.gpickle")
# G_first_gen = nx.read_gpickle("results/stability/LONG_RANDOM.gpickle")
G_first_gen = nx.read_gpickle("results/model2_generated_networks/best_of_gen_model2_0_2_DisVic_300DriveLim_900.gpickle")
# mode = 'random'
mode = 'optimum'
n = 3
if mode == 'random':
    G = G_first_gen
else:
    G = G_opt
# set_neighbors(G_opt)
# G = generate_initial_links(2, G_opt)
# nx.write_gpickle(G, 'results/stability/LONG_RANDOM.gpickle')


# Get x,y,z data from the network
x_orig = np.asarray( [G.node[node]['Longitude'] for node in G] )
y_orig = np.asarray( [G.node[node]['Latitude'] for node in G] )

tmp = {}
for node in G:
    tmp.update( {node:0} )
    # for add_docs in G.node[node]['additional_docs']:
    #     tmp[node] += add_docs

# Try a certain scenario
df_disaster = pd.read_excel('data/raw/Disaster Location.xls', index=False)
df_pop_density = pd.read_csv('data/processed/Kepadatan_Penduduk_Jakarta_Per_Kecamatan_2017.csv' )
density = []
for kec in df_disaster['nama_kecamatan']:
    density.append( df_pop_density['jumlah_kepadatan_(jiwa_per_km2)'][df_pop_density['nama_kecamatan'] == kec].iloc[0])
df_disaster['jumlah_kepadatan_(jiwa_per_km2)'] = density

df_driving_time = pd.read_csv('data/processed/Driving_Time_Matrix.csv')

list_of_disloc = list(df_dis['Location Name'][df_dis['Group']==code])
# list_of_disloc = list(df_dis['Location Name'])
# list_of_disloc = random.sample(list(df_disaster['Location Name']), 4)
run_sim.run_testing('4', 300, df_driving_time, list_of_disloc, df_disaster, G)
z_orig = np.asarray( [G.node[node]['avail_docs']+tmp[node] for node in G])

print(z_orig)
# Setup
temp_mean = 40
temp_std  = 20
debug     = False
# Setup colormap
colors = ['#d7191c',  '#fdae61',  '#ffffbf',  '#abdda4',  '#2b83ba']
vmin   = temp_mean - 2 * temp_std
vmax   = temp_mean + 2 * temp_std
levels = len(colors)
cm     = branca.colormap.LinearColormap(colors, vmin=vmin, vmax=vmax).to_step(levels)

# Make a grid
print(np.min(x_orig), np.min(y_orig))
print(G.nodes(data=True))
x_arr          = np.linspace(106.687993, 106.939920, 3000)
y_arr          = np.linspace(-6.397775, -6.1087216, 3000)
x_mesh, y_mesh = np.meshgrid(x_arr, y_arr)

# Grid the values
z_mesh = griddata((x_orig, y_orig), z_orig, (x_mesh, y_mesh), method='linear')

# Gaussian filter the grid to make it smoother
sigma = [5, 5]
z_mesh = sp.ndimage.filters.gaussian_filter(z_mesh, sigma, mode='constant')

# Create the contour
contourf = plt.contourf(x_mesh, y_mesh, z_mesh, levels, alpha=0.5, colors=colors, linestyles='None', vmin=vmin, vmax=vmax)

# Convert matplotlib contourf to geojson
geojson = geojsoncontour.contourf_to_geojson(
    contourf=contourf,
    min_angle_deg=3.0,
    ndigits=5,
    stroke_width=1,
    fill_opacity=0.5)

# Set up the folium plot
jkt_lat = -6.2205   #modified
jkt_lng = 106.8283
geomap = folium.Map([jkt_lat , jkt_lng], zoom_start=12, tiles="cartodbpositron")


# Import and Draw Jakarta borders
def my_style_function(feature):
    return {
        # 'fillColor': category_col[category],
        'color': 'black',  # line color
        'weight': 2,  # line width
        'opacity': 0.4,  # line opacity
        'fillOpacity': 0.0,  # fill color opacity
        # lambda x: {'color': 'blue', 'weight': 2, 'fillOpacity': 0.3}
    }

geo_json_data = json.load(open('data/raw/Jakarta_Districts_Info.GeoJson'))
folium.features.GeoJson(data=geo_json_data,
                   name='Jakarta',smooth_factor=2,
                style_function=my_style_function,
                  ).add_to(geomap)


# Plot the contour plot on folium
folium.GeoJson(
    geojson,
    style_function=lambda x: {
        'color':     x['properties']['stroke'],
        'weight':    x['properties']['stroke-width'],
        'fillColor': x['properties']['fill'],
        'opacity':   0.6,
    }).add_to(geomap)

# Plot Disaster
for point in range(0, len(locationlist)):
    folium.Marker( (locationlist[point][1],locationlist[point][2]), icon=folium.Icon(color='red', prefix='glyphicon',
                                        icon='glyphicon-screenshot'), popup=locationlist[point][0]).add_to(geomap)

# Plot Hospitals
for hosp in G:
    if G.node[hosp]['activated']==False:
        col = G.node[hosp]['col_node']
    else:
        col = 'purple'
    res = G.node[hosp]['avail_docs']+tmp[hosp]
    # folium.Marker((G.node[hosp]['Latitude'],G.node[hosp]['Longitude']), icon=folium.Icon(color='green', prefix='glyphicon', icon='glyphicon-plus'),
    #               popup=G.node[hosp]['NAMA RS'] + ' (' + 'Resources: ' + str(res) + ')').add_to(geomap)
    folium.Circle(
        location= (G.node[hosp]['Latitude'],G.node[hosp]['Longitude']),
        popup=G.node[hosp]['NAMA RS'] + ' (' + 'Resources: ' +
              str(res) + ')',
        radius=int(100 + 5 * res),
        color=col,
        fill=True,
        fill_color=col).add_to(geomap)

# Draw edges between connected hospital
# for node in G:
#     if G.node[node]['activated'] == False:
#         for neighbor in list(G[node]):
#             p1 = [G.node[node]['Latitude'], G.node[node]['Longitude']]
#             p2 = [G.node[neighbor]['Latitude'], G.node[neighbor]['Longitude']]
#
#             arrows = get_arrows(geomap, locations=[p1, p2], n_arrows=1)
#
#             folium.PolyLine(locations=[p1, p2], weight=1.5, opacity=1, color='black').add_to(geomap)

for edge in G.edges():
    p1 = [G.node[edge[0]]['Latitude'], G.node[edge[0]]['Longitude']]
    p2 = [G.node[edge[1]]['Latitude'], G.node[edge[1]]['Longitude']]

    # arrows = get_arrows(geomap, locations=[p1, p2], n_arrows=1)
    # folium.PolyLine(locations=[p1, p2], weight=1.0, opacity=1, color='black').add_to(geomap)

# Add the colormap to the folium map
cm.caption = 'Hospital Network - Number of doctors'
geomap.add_child(cm)

# Fullscreen mode
plugins.Fullscreen(position='topright', force_separate_button=True).add_to(geomap)

# Plot the data

if mode=='random':
    geomap.save('results/contour_plot_performance/performance_random_network_{}_model2.html'.format(n))
else:
    geomap.save('results/contour_plot_performance/performance_optimum_network_{}_model2.html'.format(n))









# p = []
# pt = []
# T = 60*5
# d = dict( G.node[3]['patient_arrival_time_counter'])
# t = 0
# while True:
#     if t in d:
#         p.append(d[t])
#         pt.append(t)
#     t+=1
#     if t==3600*2:
#         break
# counter = 0
# P = [0]
# PT = [0]
# t=0
# while True:
#     t += 1
#     if t in d:
#         counter+=d[t]
#     if t%T==0:
#         P.append(counter)
#         PT.append(int(t/60))
#         counter=0
#     if t==3600*2:
#         break
# print(d)
# print(P)
# plt.plot(PT, P)
# plt.grid()
# plt.ylabel('Number of patients arrived in 5 mins interval')
# plt.xlabel('Time (mins)')
# plt.show()
# plt.close()


#mutation from main.py
# else:
#     pass
    # print("Modified Mutation", two_previous_perf)
    # chrom_fitness = []
    #
    # for chromosome in crossed_pop:
    #     for hosp in activated_hospitals:
    #         chromosome.node[hosp]['donors_hoshostime_docs'] = []
    #
    #     perf = run_sim.calculate_performance(activated_hospitals, disaster_and_affected_hospitals,
    #                                          DISASTER_VICINITY, list_of_disloc, df_driving_time, chromosome)
    #     chrom_fitness.append( (chromosome, perf) )
    #
    # min_chrom = min(chrom_fitness, key=itemgetter(1))
    #
    # mutated_pop = []
    # # for j in range(len(crossed_pop)):
    # for j in range(1):
    #     for i in range(100):
    #         chromosome = min_chrom[0].copy()
    #         # print(chromosome.nodes(data=True))
    #         receiver = random.choice(activated_hospitals)
    #         tmp_neighbors = chromosome.node[receiver]['neighbors']
    #         if len(tmp_neighbors) == 0:
    #             print('No more choice for', receiver)
    #             continue
    #         else:
    #             while True:
    #                 node = random.choice(tmp_neighbors)
    #                 try:
    #                     if node in list(chromosome[receiver]):
    #                         continue
    #                 except:
    #                     pass
    #
    #                 if len(chromosome[node]) < MAX_EDGES:
    #                     if len(chromosome[receiver]) < MAX_EDGES:
    #                         chromosome.add_edge(node, receiver)
    #                         break
    #                     elif len(chromosome[receiver]) == MAX_EDGES:
    #                         while True:
    #                             existing_node = random.choice(list(chromosome[receiver]))
    #                             if node != existing_node:
    #                                 chromosome.remove_edge(receiver, existing_node)
    #                                 chromosome.add_edge(node, receiver)
    #                                 break
    #                         break
    #
    #                 elif len(chromosome[node]) == MAX_EDGES:
    #                     if len(chromosome[receiver]) == MAX_EDGES:
    #                         node2 = random.choice(list(chromosome[receiver]))
    #                         receiver2 = random.choice(list(chromosome[node]))
    #                         if node2 in G.node[receiver2]['neighbors']:
    #                             chromosome.remove_edge(node2, receiver)
    #                             chromosome.remove_edge(node, receiver2)
    #                             chromosome.add_edge(node, receiver)
    #                             chromosome.add_edge(node2, receiver2)
    #                             break
    #                         else:
    #                             chromosome.remove_edge(node2, receiver)
    #                             chromosome.remove_edge(node, receiver2)
    #                             chromosome.add_edge(node, receiver)
    #                             nodes2 = [x for x in G.node[receiver2]['neighbors'] if len(chromosome[x])<MAX_EDGES]
    #                             try:
    #                                 node2 = random.choice(nodes2)
    #                                 print('FORCEFUL mETHOD: ', len(chromosome[node2]),
    #                                       len(chromosome[receiver2]))
    #                                 chromosome.add_edge(node2, receiver2)
    #                             except:
    #                                 pass
    #
    #
    #         for hosp in activated_hospitals:
    #             chromosome.node[hosp]['donors_hoshostime_docs'] = []
    #         tmp_perf = run_sim.calculate_performance(activated_hospitals, disaster_and_affected_hospitals,
    #                                          DISASTER_VICINITY, list_of_disloc, df_driving_time, chromosome)
    #         if tmp_perf<min_chrom[1]:
    #             mutated_pop.append(chromosome)
    #             break
    # return crossed_pop + mutated_pop