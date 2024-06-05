#C:\Users\sage\AppData\Local\Programs\Python\Python310\python.exe .\sage_vrs.py Training_Problems\problem1.txt
import argparse
import pandas as pd
import numpy as np
import csv
from typing import Tuple
def euclidean_distance(p1: Tuple[float], p2:float) -> float:
    '''takes in two x,y points as p1 and p2'''
    #could also use numpy.linalg.norm(a-b) for vectors
    return np.sqrt((p2[0] - p1[0])**2 + (p2[1]-p1[1])**2)


def text_vrs(loads_path: str) -> None:
    '''takes in the path to a space separated txt file with a header. 
    each row of data is int Tuple(float, float) Tuple(float, float)
    representing the loadNumber pickup dropoff'''
    loads = []
    with open(loads_path, 'r') as fileobj:
        lines  =fileobj.readlines()
        for i, row in enumerate(lines):
            if i!=0:
                row = row.split(' ')
                print(row)
                loads.append([row[0], np.array(row[1], dtype=np.float64), np.array(row[2], dtype=np.float64)])
    print(loads)
def tuplestr_to_tuplefloat(x: str) -> Tuple[float]:
    x = x.strip("()")
    listx = x.split(",")
    return np.array([np.float64(listx[0]), np.float64(listx[1])])
def calculate_distance(df, possible_routes) -> float:
    new_distance=0
    origin = np.array([0,0])
    truck_location = origin
    #print()
    #print("possible routes", possible_routes)
    for li, loadNumber in enumerate(possible_routes):
        row_index = loadNumber-1
        row_data = df[df['loadNumber']==loadNumber]
        pickup = row_data.iloc[0,1]
        dropoff = row_data.iloc[0,2]
        #print('pickup', pickup, 'truck_location', truck_location)
        #print('dropoff', dropoff,'pickup', pickup)
        new_distance+= np.linalg.norm(pickup-truck_location) 
        new_distance+= np.linalg.norm(dropoff-pickup)
        truck_location = dropoff
    #print("end", origin,"truck_location", truck_location )
    new_distance += np.linalg.norm(origin - truck_location)
    #print(new_distance)     
    return new_distance    
def vrs(loads_path: str) -> None:
    '''takes in the path to a space separated txt file with a header. 
    each row of data is int Tuple(float, float) Tuple(float, float)
    representing the loadNumber pickup dropoff'''
    df = pd.read_csv(loads_path, sep=' ', dtype={'loadNumber':np.int16, 'pickup':str, 'dropoff':str})
    df['pickup'] = df['pickup'].apply(tuplestr_to_tuplefloat)
    df['dropoff'] = df['dropoff'].apply(tuplestr_to_tuplefloat)
    drivers = []
    curr_driver = 0
    for dfi, row in df.iterrows():
        #driving_dist = np.linalg.norm(row.iloc[2]-row.iloc[1]) + np.linalg.norm(row.iloc[1])
        #print("route", row.iloc[0], "numpy.linalg.norm(a-b)",driving_dist )
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
if __name__=='__main__':
    parser = argparse.ArgumentParser(prog="vehicle_routing_solver" , description = "Pass in the path to the txt file with the loads to pickup", 
                                    )
    parser.add_argument('loads_path', help="path to the txt file with loads to pickup") 
    args = parser.parse_args()
    vrs(args.loads_path)