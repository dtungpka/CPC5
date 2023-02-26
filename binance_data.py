import requests
import datetime
import pandas as pd
import os
import shutil
import json
import numpy as np
import matplotlib.pyplot as plt
import time
import threading
details = json.loads(open("details.json", "r").read())

# sample url: https://data.binance.vision/data/futures/um/daily/klines/DOGEUSDT/5m/DOGEUSDT-5m-2023-02-25.zip


process_bars = [0,0]
def unpack_data(path):
    """Unpacks data from a given csv file path, return a pandas dataframe"""
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    return df
def get_csv(url):
    """Downloads a csv file from a given url, unzip it, return path to csv and path to zip"""
    if not os.path.exists("data"):
        os.mkdir("data")
    if not os.path.exists("data/zip"):
        os.mkdir("data/zip")
    if not os.path.exists("data/csv"):
        os.mkdir("data/csv")
    zip_path = "data/zip/" + url.split("/")[-1]
    csv_path = "data/csv/" + url.split("/")[-1].replace(".zip", ".csv")
    if not os.path.exists(csv_path):
        try:
            r = requests.get(url)
        except:
            return None, None
        with open(zip_path, "wb") as f:
            f.write(r.content)
        shutil.unpack_archive(zip_path, "data/csv")

    return csv_path, zip_path

def get_date_range(start, end):
    """Returns a yield of dates between start and end"""
    start = datetime.datetime.strptime(start, "%Y-%m-%d")
    end = datetime.datetime.strptime(end, "%Y-%m-%d")
    while start <= end:
        yield start.strftime("%Y-%m-%d")
        start += datetime.timedelta(days=1)

def get_data(coin: str, interval:str, start, end,process_ID=0,output="data.csv"):
    """Continuously downloads data from binance data api, writes to csv file"""
    global process_bars
    output = coin + "_" + interval + "_" + output
    date_count = datetime.datetime.strptime(end, "%Y-%m-%d") - datetime.datetime.strptime(start, "%Y-%m-%d")
    for date in get_date_range(start, end):
        url = f"https://data.binance.vision/data/futures/um/daily/klines/{coin}/{interval}/{coin}-{interval}-{date}.zip"
        #set process bar
        process_bars[process_ID] = (datetime.datetime.strptime(date, "%Y-%m-%d") - datetime.datetime.strptime(start, "%Y-%m-%d")).days / date_count.days
        while True:
            csv_path, zip_path = get_csv(url)
            if csv_path:
                df = unpack_data(csv_path)
                df.to_csv(output, mode="a", header=False)
                #os.remove(csv_path)
                if os.path.exists(zip_path):
                    os.remove(zip_path)
                break
                #print(f"Downloaded data for {date}")
            else:
                #print(f"Could not download data for {date}")
                time.sleep(1)
                


if __name__ == "__main__":
    args1 = (details["coin"], details["interval"], details["start"], details["end"],0)
    args2 = (details["coin"], details["prediction"], details["start"], details["end"],1)
    thr1 = threading.Thread(target=get_data, args=args1)
    thr2 = threading.Thread(target=get_data, args=args2)
    thr1.start()
    thr2.start()
    exist = []
    #check for existing data in current directory
    for file in os.listdir():
        if file.endswith(".csv"):
            exist.append(file)
    if len(exist) > 0:
        print("Found existing data files: ",exist)
        print("Do you want to delete them? (y/n)")
        while True:
            choice = input()
            if choice == "y":
                for file in exist:
                    os.remove(file)
                break
            elif choice == "n":
                break
            else:
                print("Invalid input")
    #print progress
    while thr1.is_alive() or thr2.is_alive():
        print("\rDownloading data: ",round(process_bars[0]*100),"%",round(process_bars[1]*100),"%",end="")
        time.sleep(1)
    #loop through files csv, check if they are empty or not exist, print error
    empty = 0
    for date in get_date_range(details["start"], details["end"]):
        csv_path = "data/csv/" + details["coin"] + "-" + details["interval"] + "-" + date + ".csv"
        if not os.path.exists(csv_path):
            print("Error: ",csv_path," does not exist")
            empty
        else:
            df = unpack_data(csv_path)
            if df.empty:
                print("Error: ",csv_path," is empty")
                empty += 1
    for date in get_date_range(details["start"], details["end"]):
        csv_path = "data/csv/" + details["coin"] + "-" + details["prediction"] + "-" + date + ".csv"
        if not os.path.exists(csv_path):
            print("Error: ",csv_path," does not exist")
            empty
        else:
            df = unpack_data(csv_path)
            if df.empty:
                print("Error: ",csv_path," is empty")
                empty += 1
    if empty == 0:
        print("\nData downloaded successfully")

    


