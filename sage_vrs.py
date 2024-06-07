#C:\Users\sage\AppData\Local\Programs\Python\Python310\python.exe .\sage_vrs.py Training_Problems\problem1.txt
import argparse
import pandas as pd
import numpy as np
import csv
from typing import Tuple, List
from scipy.spatial.distance import pdist, squareform
import networkx as nx
from vrpy import VehicleRoutingProblem
import logging
from time_region import timer_func, time_region
logging.getLogger("vrpy").setLevel(logging.ERROR)
def euclidean_distance(p1: Tuple[float], p2:float) -> float:
    '''takes in two x,y points as p1 and p2'''
    #could also use numpy.linalg.norm(a-b) for vectors
    return np.sqrt((p2[0] - p1[0])**2 + (p2[1]-p1[1])**2)

@timer_func
def tuplestr_to_tuplefloat(x: str) -> Tuple[float]:
    x = x.strip("()")
    listx = x.split(",")
    return np.array([np.float64(listx[0]), np.float64(listx[1])])

@timer_func
def calculate_distance(df, possible_routes) -> float:
    new_distance=0
    origin = np.array([0,0])
    truck_location = origin
    for li, loadNumber in enumerate(possible_routes):
        row_data = df[df['loadNumber']==loadNumber]
        pickup = row_data.iloc[0,1]
        dropoff = row_data.iloc[0,2]
        new_distance+= np.linalg.norm(pickup-truck_location) 
        new_distance+= np.linalg.norm(dropoff-pickup)
        truck_location = dropoff
    new_distance += np.linalg.norm(origin - truck_location)
    return new_distance    
@timer_func
def load_data(loads_path: str) -> pd.DataFrame:
    df = pd.read_csv(loads_path, sep=' ', dtype={'loadNumber':np.int16, 'pickup':str, 'dropoff':str})
    df['pickup'] = df['pickup'].apply(tuplestr_to_tuplefloat)
    df['dropoff'] = df['dropoff'].apply(tuplestr_to_tuplefloat)    
    return df

@timer_func
def vrs_forloop_build_route(df: pd.DataFrame=None) -> List:
    drivers = []
    curr_driver = 0
    for dfi, row in df.iterrows():
        ##drive only one route
        if curr_driver==0:
            drivers.append([row.iloc[0]])
            curr_driver = np.linalg.norm(row.iloc[2]-row.iloc[1]) + np.linalg.norm(row.iloc[1]) + np.linalg.norm(row.iloc[2])
        else:
            possible_routes = drivers[-1]+ [ row.iloc[0]]
            new_distance=calculate_distance(df, possible_routes)
            if new_distance < 12*60:
                drivers[-1].append(row.iloc[0])
                curr_driver=new_distance
            else:
                drivers.append([row.iloc[0]])
                curr_driver= np.linalg.norm(row.iloc[2]-row.iloc[1]) + np.linalg.norm(row.iloc[1]) + np.linalg.norm(row.iloc[2])
    return drivers
@timer_func
def vrs_forloop(loads_path: str=None, df: pd.DataFrame=None, sort_angle=True) -> None:
    '''this vrs implementation will not produce the most efficient solution. It simply produces a solution'''
    if not loads_path and not df:
        raise Exception("either loads_path or df are required")
    if loads_path:
        df = load_data(loads_path)
    if sort_angle:
        df['radial_angle'] = df.apply(lambda x: np.arctan2(x.pickup[1], x.pickup[0]), axis=1)
        df.sort_values('radial_angle', inplace=True)
    drivers = vrs_forloop_build_route(df)
    for route in drivers:
        print(route)

@timer_func
def create_graph(distance_matrix: np.ndarray, pickup_dropoff: dict):
    graph_cost =nx.from_numpy_array(np.array(distance_matrix, dtype=[("cost", float)]), create_using=nx.DiGraph)
    graph_time =nx.from_numpy_array(np.array(distance_matrix, dtype=[("time", float)]), create_using=nx.DiGraph)
    graph = nx.compose(graph_cost, graph_time)
    graph = nx.relabel_nodes(graph, {0: "Source", len(distance_matrix)-1: "Sink"})
    for (pickup_i, dropoff_i) in pickup_dropoff:
        graph.nodes[pickup_i]["request"] = dropoff_i
        graph.nodes[pickup_i]["demand"] = 1
        graph.nodes[dropoff_i]["demand"] = -1
    return graph

@timer_func
def calc_distances(df: pd.DataFrame) -> np.ndarray:
    all_addresses = np.concatenate([[np.array([0,0])], np.vstack(pd.concat([df['pickup'], df['dropoff']]).values), [np.array([0,0])]],axis=0)
    distance_matrix = squareform(pdist(all_addresses, metric='euclidean')).astype(np.int32)
    #print("distance_matrix",distance_matrix.shape)
    distance_matrix[:,0] = 0
    distance_matrix[-1,:] = 0
    return distance_matrix

@timer_func
def vrs_vrpy(loads_path: str) -> None:
    '''this solution uses the vrpy library to optimize the route'''
    ##referenced https://medium.com/@trentleslie/leveraging-the-vehicle-route-problem-with-pickup-and-dropoff-vrppd-for-optimized-beer-delivery-in-392117d69033
    df = load_data(loads_path)
    distance_matrix=calc_distances(df)
    #print(distance_matrix[:4,:4])
    #print(distance_matrix[-4:,-4:])
    pickup_dropoff = {(x+1,len(df)+x+1):1 for x in range(len(df))}
    graph = create_graph(distance_matrix, pickup_dropoff)
    prob = VehicleRoutingProblem(graph, load_capacity=1)
    prob.pickup_delivery = True
    prob.duration = 12*60
    prob.fixed_cost = 500
    sol = prob.solve(time_limit=20, cspy=False, solver='cbc', pricing_strategy ='Exact')
    #print("sol", sol)
    print("best routes", prob.best_routes)
    #print("best value",prob.best_value)
    if prob.best_value:
        for route in prob.best_routes.values():
            print([x for x in route if (x not in ['Source', 'Sink'] and x <= len(df))])
    else:
        vrs_forloop(df=df)

@timer_func
def vrs_initialized(loads_path: str) -> None:
    '''use the vrpy package but initialize it with the for-loop algo'''
    df = load_data(loads_path)
    drivers = vrs_forloop_build_route(df)
    initial_routes = []
    for i in range(len(drivers)):
        curr_route = ['Source'] 
        for x in drivers[i]:
            curr_route.append(x)    
            curr_route.append(x+len(df))    
        curr_route.append("Sink")
        initial_routes.append(curr_route)
    distance_matrix=calc_distances(df)
    pickup_dropoff = {(x+1,len(df)+x+1):1 for x in range(len(df))}
    graph = create_graph(distance_matrix, pickup_dropoff)
    prob = VehicleRoutingProblem(graph, load_capacity=1)
    prob.pickup_delivery = True
    prob.duration = 12*60
    prob.fixed_cost = 500
    #print(initial_routes)
    #initial_routes = {1: ['Source', 1, 11, 'Sink'], 2: ['Source', 2, 12, 'Sink'], 3: ['Source', 3, 13, 'Sink'], 4: ['Source', 6, 16, 'Sink'], 5: ['Source', 7, 17, 'Sink'], 6: ['Source', 8, 18, 'Sink'], 7: ['Source', 10, 20, 'Sink'], 8: ['Source', 4, 14, 5, 15, 9, 19, 'Sink']}
    sol = prob.solve(initial_routes= initial_routes,time_limit=20, cspy=False, solver='cbc', pricing_strategy ='Exact')
    #print("best routes", prob.best_routes)
    if prob.best_value:
        for route in prob.best_routes.values():
            print([x for x in route if (x not in ['Source', 'Sink'] and x <= len(df))])
    else:
        for route in drivers:
            print(route)


@timer_func
def vrs_nearest_next(loads_path: str=None, df: pd.DataFrame=None) -> None:
    '''build off the previous for loop design but grab the next nearest starting point'''
    if not loads_path and not df:
        raise Exception("either loads_path or df are required")
    if loads_path:
        df = load_data(loads_path)
    drivers = []
    curr_driver = 0
    distance_matrix=calc_distances(df)
    need_pickup = np.ones(len(df))
    current_node = 0
    while need_pickup.sum() >0:
        possible_next_node = np.argmin(distance_matrix[current_node,1:1+len(df)])
        print("from", current_node, "next", possible_next_node)
        if curr_driver==0:
            drivers.append([row.iloc[0]])
            curr_driver = np.linalg.norm(df.iloc[2]-row.iloc[1]) + np.linalg.norm(row.iloc[1]) + np.linalg.norm(row.iloc[2])

    for dfi, row in df.iterrows():
        ##drive only one route
        if curr_driver==0:
            drivers.append([row.iloc[0]])
            curr_driver = np.linalg.norm(row.iloc[2]-row.iloc[1]) + np.linalg.norm(row.iloc[1]) + np.linalg.norm(row.iloc[2])
        else:
            possible_routes = drivers[-1]+ [ row.iloc[0]]
            new_distance=calculate_distance(df, possible_routes)
            if new_distance < 12*60:
                drivers[-1].append(row.iloc[0])
                curr_driver=new_distance
            else:
                drivers.append([row.iloc[0]])
                curr_driver= np.linalg.norm(row.iloc[2]-row.iloc[1]) + np.linalg.norm(row.iloc[1]) + np.linalg.norm(row.iloc[2])

    for route in drivers:
        print(route)


def vrs(loads_path: str, mode:str='for-loop') -> None:
    '''takes in the path to a space separated txt file with a header. 
    each row of data is int Tuple(float, float) Tuple(float, float)
    representing the loadNumber pickup dropoff'''    
    if mode=='vrpy':
        vrs_vrpy(loads_path)
    elif mode=='for-loop':
        vrs_forloop(loads_path)
    elif mode=='initialized':
        vrs_initialized(loads_path)
if __name__=='__main__':
    parser = argparse.ArgumentParser(prog="vehicle_routing_solver" , description = "Pass in the path to the txt file with the loads to pickup", 
                                    )
    parser.add_argument('loads_path', help="path to the txt file with loads to pickup") 
    parser.add_argument('-m', '--mode', help='which solver algorithm to use: for-loop, initialized, or vrpy',
                        default='for-loop')
    args = parser.parse_args()
    vrs(args.loads_path, mode=args.mode)
    #time_region.log_summary()