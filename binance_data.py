
import requests
import datetime
import pandas as pd
import os
import shutil
import json
import numpy as np
import time
import threading
import re
from tqdm import tqdm
details = json.loads(open("details.json", "r").read())
current_coin = ""
# sample url: https://data.binance.vision/data/futures/um/daily/klines/DOGEUSDT/5m/DOGEUSDT-5m-2023-02-25.zip
MAX_TRY = 5
update = False
process_bars = [0,0]
TOTALS = [0,0]
def unpack_data(path):
    """Unpacks data from a given csv file path, return a pandas dataframe"""
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    return df
def get_csv(url):
    """Downloads a csv file from a given url, unzip it, return path to csv and path to zip"""
    zip_path = "data/"+current_coin+"/zip/" + url.split("/")[-1]
    csv_path = "data/"+current_coin+"/csv/" + url.split("/")[-1].replace(".zip", ".csv")
    trys = 0
    while not os.path.exists(csv_path) and trys <= MAX_TRY:
        try:
            r = requests.get(url)
        except:
            trys += 1
            continue
        with open(zip_path, "wb+") as f:
            f.write(r.content)
            f.close()
        try:
            shutil.unpack_archive(zip_path, "data/"+current_coin+"/csv")
        except:
            trys += 1
            continue
    if trys >= MAX_TRY:
        return None, None
    return csv_path, zip_path

def get_date_range(start, end):
    """Returns a yield of dates between start and end"""
    if type(start) is str:
        start = datetime.datetime.strptime(start, "%Y-%m-%d")
    end = datetime.datetime.strptime(end, "%Y-%m-%d")
    while start <= end:
        yield start.strftime("%Y-%m-%d")
        start += datetime.timedelta(days=1)
def get_start_date(url):
    #get html
    end_d = None
    start_d = datetime.datetime.now()
    c = 0
    while True:
        r = str(requests.get(url).content)
        #find all occurrences of .CHECKSUM
        #1000BTTCUSDT-15m-2022-04-05.zip.CHECKSUM
        CKs = [m.start() for m in re.finditer('.CHECKSUM', r)]
        if len(CKs) == 1:
            break
        for i in range(len(CKs)):
            try:
                #extract the date before i
                date = r[CKs[i]-14:CKs[i]-4] #'2022-04-05'
                #check if date is valid
                d = datetime.datetime.strptime(date, '%Y-%m-%d')
                if d < start_d:
                    start_d = d
                if end_d == None or end_d < d:
                    end_d = d
            except:
                pass
        nd = (end_d + datetime.timedelta(10 * c)).strftime("%Y-%m-%d") #2023-02-09
        url = "https://s3-ap-northeast-1.amazonaws.com/data.binance.vision?delimiter=/&prefix=data/futures/um/daily/klines/"+current_coin+"/"+details['interval']+"/&marker=data%2Ffutures%2Fum%2Fdaily%2Fklines%2F"+current_coin+"%2F"+details['interval']+"%2F"+current_coin+"-"+details['interval']+"-"+nd+".zip.CHECKSUM"
        c += 1
    return start_d, end_d > (datetime.datetime.strptime(details["end"], '%Y-%m-%d') - datetime.timedelta(2))
        
def get_data(coin: str, interval:str, start, end,process_ID=0,output="data.csv"):
    """Continuously downloads data from binance data api, writes to csv file"""
    global process_bars,TOTALS
    output = "downloaded/"+coin + "_" + interval + "_" + output
    
    
    date_count = datetime.datetime.strptime(end, "%Y-%m-%d") - start
    TOTALS[process_ID] = date_count.days
    for date in get_date_range(start, end):
        url = f"https://data.binance.vision/data/futures/um/daily/klines/{coin}/{interval}/{coin}-{interval}-{date}.zip"
        #set process bar
        process_bars[process_ID] = (datetime.datetime.strptime(date, "%Y-%m-%d") - start).days
        
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
                


def download_data():
    u_ = f"https://s3-ap-northeast-1.amazonaws.com/data.binance.vision?delimiter=/&prefix=data/futures/um/daily/klines/"+current_coin+"/"+details['interval']+"/"
    start,r =get_start_date(u_)
    if (r == False):
        print("Coin discontinued, skipping..")
        return
    if not os.path.exists("data"):
        os.mkdir("data")
    if not os.path.exists("downloaded"):
        os.mkdir("downloaded")
    if not os.path.exists("data/"+current_coin):
        os.mkdir("data/"+current_coin)
    if not os.path.exists("data/"+current_coin+"/zip"):
        os.mkdir("data/"+current_coin+"/zip")
    if not os.path.exists("data/"+current_coin+"/csv"):
        os.mkdir("data/"+current_coin+"/csv")
    args1 = (current_coin, details["interval"], start, details["end"],0)
    args2 = (current_coin, details["prediction"], start, details["end"],1)
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
    t = TOTALS[0] + TOTALS[1]
    with tqdm(total=t) as pbar:
        while thr1.is_alive() or thr2.is_alive():
            t_ = process_bars[0] + process_bars[1]
            pbar.update(t_ - pbar.n)  # Update progress bar with difference between current value of t and progress bar's current value
            time.sleep(0.1)
        
    #loop through files csv, check if they are empty or not exist, print error
    empty = 0
    for date in get_date_range(details["start"], details["end"]):
        csv_path = "data/"+current_coin+"/csv/" + current_coin + "-" + details["interval"] + "-" + date + ".csv"
        if not os.path.exists(csv_path):
            empty += 1
        else:
            df = unpack_data(csv_path)
            if df.empty:
                
                empty += 1
    for date in get_date_range(details["start"], details["end"]):
        csv_path = "data/"+current_coin+"/csv/" + current_coin + "-" + details["prediction"] + "-" + date + ".csv"
        if not os.path.exists(csv_path):
           empty += 1
            
        else:
            df = unpack_data(csv_path)
            if df.empty:
                
                empty += 1
    if empty == 0:
        print("\nData downloaded successfully\n")
    else:
        #rename the "downloaded/"+coin + "_" + interval + "_" + output file to ~
        fn = "downloaded/"+current_coin + "_" + details["interval"] + "_data.csv"
        if os.path.exists(fn):
            os.rename(fn,fn+"~")
        fn = "downloaded/"+current_coin + "_" + details["prediction"] + "_data.csv"
        if os.path.exists(fn):
            os.rename(fn,fn+"~")
        print("\nData downloaded with errors\n")

if __name__ == "__main__":
    coins = []
    fn = "coin"
    if (input("Nhap ten file: 1/2? >") == "1"):
        fn += "1.txt"
    else:
        fn += "2.txt"
    with open(fn,'r+') as f:
        for line in f:
            coins.append(line.strip().replace(" ",""))
    to_write = coins.copy()
    
    for i in range(len(coins)):
        
        current_coin = coins[i]
        print("Downloading data for",current_coin)
        download_data()
    #check for [coin]_5m_data.csv and [coin]_15m_data.csv in downloaded folder and delete it from coins.txt
        for file in os.listdir("downloaded"):
            if (file.endswith("_5m_data.csv") or file.endswith("_15m_data.csv") ):
               if file.split("_")[0] in to_write:    
                to_write.remove(file.split("_")[0])
        #write coins to coins.txt
        f = open(fn,'w+')
        for coin in to_write:
            f.write(coin + "\n")
            f.flush()
        f.close()
        

        

    


