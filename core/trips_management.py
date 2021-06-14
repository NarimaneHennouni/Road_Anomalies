import os 
from datetime import datetime

def trip_name_to_numbers(file):
    file = file[:-4]
    file_split = file.split('_')
    print(file_split[1])
    return {
        'name': file_split[0],
        "date" : datetime.strptime(file_split[1], '%Y%m%d%H%M%S')
    } 

def get_file_name(trip):
    return  trip['name'] + trip['date'].strftime('%Y%m%d%H%M%S') + '.csv'

def read_all_trips(TRIPS_FOLDER):
    results = [] 
    for file in os.listdir(TRIPS_FOLDER):
        if file.endswith(".csv"):
            results.append(trip_name_to_numbers(file))
    return results