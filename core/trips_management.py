import os 
from datetime import datetime

def trip_name_to_numbers(file):
    file = file[:-4]
    file_split = file.split('_')
    return {
        'id' : file,
        'name': file_split[0],
        "date" : datetime.strptime(file_split[1], '%Y%m%d%H%M%S')
    } 

def get_file_name(name, date):
    return  name + '_' +date.strftime('%Y%m%d%H%M%S') + '.csv'

def read_all_trips(TRIPS_FOLDER):
    results = [] 
    for file in os.listdir(TRIPS_FOLDER):
        if file.endswith(".csv"):
            results.append(trip_name_to_numbers(file))
    return results

def create_trip(TRIPS_FOLDER, name):
    date_ = datetime.now()
    file_name = get_file_name(name, date_)
    with open(TRIPS_FOLDER + '/' + file_name, "w") as _:
        pass
    return {
        'id' : file_name[:-4],
        'name': name,
        "date" : date_
    }

def delete_trip(TRIPS_FOLDER, id):
    try:
        os.remove(TRIPS_FOLDER + '/' + id + '.csv')
        return True
    except OSError:
        return False

def get_stats(TRIPS_FOLDER, id):
    pass
    
    