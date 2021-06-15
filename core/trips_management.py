import os 
from datetime import datetime
import csv
import pandas as pd
from flask import send_file

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
    results = {}
    for file in os.listdir(TRIPS_FOLDER):
        if file.endswith(".csv"):
            res = trip_name_to_numbers(file)
            results[res['id']] = res
    return results

def create_trip(TRIPS_FOLDER, name):
    date_ = datetime.now()
    file_name = get_file_name(name, date_)
    with open(TRIPS_FOLDER + '/' + file_name, "w") as f:
        writer = csv.writer(f)
        writer.writerow(('class', 'cor_1', 'cor_2', 'cor_3', 'cor_4', 'lat', 'long'))

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
    try:
        df = pd.read_csv(TRIPS_FOLDER + '/' + id + '.csv')  
        return list(df.T.to_dict().values())  
    except FileNotFoundError: 
        return False
   
def get_image(IMAGES_FOLDER, id, detection_number):
    filename = IMAGES_FOLDER + '/' + id + '/' + detection_number + '.jpg'
    try:
        return send_file(filename, mimetype='detection/image')
    except FileNotFoundError:
        return False
      
    