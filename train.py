import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import json
import time
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
import itertools
import random
MAX_PERIOD = 10
FEATURES = ['OPEN', 'HIGH', 'LOW', 'CLOSE'
            ,'VOLUME','SMA','EMA','RSI',
            'BOIL','MACD','OBV','ADX','ATR','FIBO']
FEATURE = ['OPEN', 'HIGH', 'LOW', 'CLOSE'
            ,'VOLUME','SMA','EMA','RSI',
            'BOILINGER_UP','BOILINGER_DOWN','MACD','OBV','ADX','ATR','FIBO0','FIBO1','FIBO2','FIBO3','FIBO4']
BOIL = ['BOILINGER_UP','BOILINGER_DOWN']
FIBO = ['FIBO0','FIBO1','FIBO2','FIBO3','FIBO4']
def get_combination(min_len):
    # combination = []
    # for i in range(min_len,len(FEATURES)):
    #     for c in itertools.combinations(FEATURES,i):
    #         combination.append(c)
    # #shuffle the combination
    # random.shuffle(combination)
    # #trim the combination
    # combination = combination[:20]
    # combination.append(FEATURES)
    # #if BOIL in combination: remove BOIL and add BOIL_UP and BOIL_DOWN
    # for i in range(len(combination)):
    #     if 'BOIL' in combination[i]:
    #         combination[i] = list(combination[i])
    #         combination[i].remove('BOIL')
    #         combination[i].extend(BOIL)
    # #if FIBO in combination: remove FIBO and add FIBO0, FIBO1, FIBO2, FIBO3, FIBO4
    # for i in range(len(combination)):
    #     if 'FIBO' in combination[i]:
    #         combination[i] = list(combination[i])
    #         combination[i].remove('FIBO')
    #         combination[i].extend(FIBO)

    return [['VOLUME', 'EMA', 'BOILINGER_UP', 'BOILINGER_DOWN', 'OBV', 'ADX', 'FIBO0', 'FIBO1', 'FIBO2', 'FIBO3']]

class KNearestNeighbor():
    def __init__(self):
        pass

    def train(self, X, y):
        self.X_train = X
        self.y_train = y

    def predict(self, X, distance_function, predict_function, k=1):
        dists  = distance_function(self.X_train, X)
        labels = predict_function(self.y_train, dists, k=k)
        return labels
    
    @staticmethod
    def evaluate(y_test_pred, y_test, isprint=True):
        num_test = len(y_test.flatten())
        num_correct = np.sum(y_test_pred.flatten() == y_test.flatten())
        accuracy = float(num_correct) / num_test
        if isprint:
            print('Got %d / %d correct => accuracy: %f' % (num_correct, num_test, accuracy))
            #print the TP, TN, FP, FN
            #loop and calcluate with each label
            for i in range(5):
                TP = np.sum((y_test_pred == i) & (y_test == i))
                TN = np.sum((y_test_pred != i) & (y_test != i))
                FP = np.sum((y_test_pred == i) & (y_test != i))
                FN = np.sum((y_test_pred != i) & (y_test == i))
                print('label %d: TP %d, TN %d, FP %d, FN %d' % (i, TP, TN, FP, FN))

            
        return accuracy
def compute_distances_no_loop(X_train, X_test):
    result_mtx = np.sqrt(np.sum(X_test**2, axis=1, keepdims=True) + np.sum(X_train**2, axis=1) - 2 * X_test.dot(X_train.T))
    
    return result_mtx

def visualize_distance(dists):
    plt.imshow(dists, interpolation='none')
    plt.colorbar()
    plt.show()
def predict_labels(y_train, dist,num_test, k=1):
    result_mtx = np.argsort(dist, axis=1)[:, :k]
    result = np.zeros(result_mtx.shape[0])
    for i in range(num_test):
        result[i] = np.argmax(np.bincount(y_train[result_mtx[i]].flatten()))
    return result
class KNN:
    def __init__(self,coin,path:str) -> None:
        self.path = path
        self.coin = coin
    def load_data(self,path):
        self.details = json.loads(open("details.json", "r").read())

        self.X_data_df = pd.read_csv(path)
        Y_data_raw = pd.read_csv(path)
        #get only columns: open,	high,	low,	close
        self.X_data_df = self.X_data_df.iloc[:,2:12]
        Y_data_raw = Y_data_raw.iloc[:,2:12]

        #show data with each column is a different color
        self.X_data_df.iloc[:,0].plot(subplots=True, figsize=(20,10))
        #to numpy 
        self.X_data_raw = self.X_data_df.to_numpy()
        self.Y_data_raw = Y_data_raw.to_numpy()
    def true_labels(self):
        labels = self.details['label']
        #open + close / 2
        Y_data = (self.Y_data_raw[:,0] + self.Y_data_raw[:,3]) / 2
        # Y_data[i] - Y_data[i-1]
        Y_data_diff = np.diff(Y_data)
        #calculate the percentage change
        Y_data = Y_data_diff / Y_data[1:]
        #Using k-means to cluster the data into 5 groups, scatter plot
        kmeans = KMeans(n_clusters=5, random_state=0).fit(Y_data.reshape(-1,1))
        self.Y = kmeans.labels_
    def get_features(self,period = 96*5):
        X_data_raw = self.X_data_raw
        X_data_df = self.X_data_df
        PERIOD = period
        #Open + High + Low + Close + Volume
        X = np.zeros((X_data_raw.shape[0] - PERIOD, 19))
        #First 5
        X[:,0:5] = X_data_raw[PERIOD:,0:5]

        #normalize Volume
        X[:,4] = X_data_raw[PERIOD:,4] / X_data_raw[PERIOD:,4].max()

        #SMA in span of 1 day. (96 point of data)
        sma = X_data_df.iloc[:,3].rolling(window=PERIOD).mean()[PERIOD:]
        X[:,5] = sma

        #EMA in span of 1 day. (96 point of data)
        #EMA on high
        ema = X_data_df.iloc[:,1].ewm(span=PERIOD).mean()[PERIOD:]
        X[:,6] = ema

        #RSI in span of 1 day. (96 point of data)
        delta = X_data_df.iloc[:,3].diff()
        A = delta.rolling(PERIOD).mean()

        # Calculate average gain and average loss
        AG = np.where(delta > 0, abs(A), 0 )
        AL =  np.where(delta < 0, abs(A), 1 )

        # Calculate RS and RSI
        RS = AG / AL
        RSI = 100 - (100 / (1 + RS))
        X[:,7] = RSI[PERIOD:]
        #Boilinger Bands
        #Calculate the rolling mean and standard deviation
        rolling_mean = X_data_df.iloc[:,3].rolling(window=PERIOD).mean()[PERIOD:]
        rolling_std = X_data_df.iloc[:,3].rolling(window=PERIOD).std()[PERIOD:]

        #Calculate the upper and lower Bollinger Bands
        upper_band = rolling_mean + (rolling_std * 2)
        lower_band = rolling_mean - (rolling_std * 2)
        X[:,8] = upper_band
        X[:,9] = lower_band

        #MACD
        #Calculate the short term exponential moving average (EMA)
        ShortEMA = X_data_df.iloc[:,3].ewm(span=12, adjust=False).mean()[PERIOD:]

        #Calculate the long term exponential moving average (EMA)
        LongEMA = X_data_df.iloc[:,3].ewm(span=26, adjust=False).mean()[PERIOD:]

        #Calculate the MACD line
        MACD = ShortEMA - LongEMA
        X[:,10] = MACD

        #OBV
        #Calculate the On Balance Volume
        OBV = np.zeros(X_data_raw.shape[0] - PERIOD)
        OBV[0] = X_data_raw[PERIOD,4]
        for i in range(1, X_data_raw.shape[0] - PERIOD):
            if X_data_raw[i + PERIOD,3] > X_data_raw[i + PERIOD - 1,3]:
                OBV[i] = OBV[i-1] + X_data_raw[i + PERIOD,4]
            elif X_data_raw[i + PERIOD,3] < X_data_raw[i + PERIOD - 1,3]:
                OBV[i] = OBV[i-1] - X_data_raw[i + PERIOD,4]
            else:
                OBV[i] = OBV[i-1]
        X[:,11] = OBV

        #normalize OBV
        X[:,11] = X[:,11] / X[:,11].max()


        #ADX
        #Calculate the True Range
        TR = np.zeros(X_data_raw.shape[0])
        for i in range(1, X_data_raw.shape[0] ):
            TR[i] = max(X_data_raw[i ,1] - X_data_raw[i ,2], abs(X_data_raw[i ,1] - X_data_raw[i  - 1,3]), abs(X_data_raw[i ,2] - X_data_raw[i  - 1,3]))

        #Calculate the Positive Directional Indicator and Negative Directional Indicator
        PDI = np.zeros(X_data_raw.shape[0] )
        NDI = np.zeros(X_data_raw.shape[0] )
        for i in range(1, X_data_raw.shape[0] ):
            PDI[i] = 100 * (X_data_raw[i ,1] - X_data_raw[i  - 1,3]) / TR[i]
            NDI[i] = 100 * (X_data_raw[i  - 1,3] - X_data_raw[i ,2]) / TR[i]

        #Calculate the Average Directional Movement Index
        PDI = pd.DataFrame(PDI).ewm(span=14).mean()
        NDI = pd.DataFrame(NDI).ewm(span=14).mean()
        DX = 100 * abs(PDI - NDI) / (PDI + NDI)
        ADX = pd.DataFrame(DX).ewm(span=14).mean()
        X[:,12] = ADX[PERIOD:].to_numpy().reshape(X.shape[0])

        #normalize ADX
        X[:,12] = X[:,12] / X[:,12].max()

        #ATR
        #Use the above TR
        ATR = pd.DataFrame(TR).ewm(span=14).mean()
        X[:,13] = ATR[PERIOD:].to_numpy().reshape(X.shape[0])
        #Fibonacci Retracement
        #Calculate the Fibonacci Retracement
        Fibonacci = np.zeros((X_data_raw.shape[0] - PERIOD, 5))
        Fibonacci[:,0] = X_data_raw[PERIOD:,3] - X_data_raw[PERIOD:,2]
        Fibonacci[:,1] = X_data_raw[PERIOD:,1] - X_data_raw[PERIOD:,3]
        Fibonacci[:,2] = X_data_raw[PERIOD:,1] - X_data_raw[PERIOD:,2]
        Fibonacci[:,3] = X_data_raw[PERIOD:,3] - X_data_raw[PERIOD:,0]
        Fibonacci[:,4] = X_data_raw[PERIOD:,1] - X_data_raw[PERIOD:,0]

        #normalize Fibonacci
        Fibonacci = Fibonacci / Fibonacci.max()

        X[:,14:19] = Fibonacci
        return X
    def train(self,X,Y,p,c:list):
        #gete the index of c in FEATURES
        c_index = []
        for i in range(len(c)):
            c_index.append(FEATURE.index(c[i]))
        #Filter the X columns based on c
        X = X[:,c_index]






        X = X[1:]
        Y = Y[p:]
        l = X.shape[0]
        ratio = 0.4
        #only use the first 80% of the data for training
        X = X[1:int(l*ratio)]
        Y = Y[1:int(l*ratio)]
        #Validation set is the rest of the data
        #self.X_val = X[int(l*ratio):]
        #self.Y_val = Y[int(l*ratio):]
        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
        classifier = KNearestNeighbor()
        classifier.train(X_train, y_train)
        num_test = X_test.shape[0]
        dists_no = compute_distances_no_loop(X_train, X_test)
        y_pred = predict_labels(y_train, dists_no,num_test, k=1)
        print("Test Accuracy: ")
        acc = classifier.evaluate(y_pred, y_test)

        #the validation set is the rest of the data
        #print("Validation Accuracy: ")
        #dists_no = compute_distances_no_loop(X_train, self.X_val)
        #y_pred = predict_labels(y_train, dists_no,self.X_val.shape[0], k=1)
        #acc_val = classifier.evaluate(y_pred, self.Y_val)
        return 
    def update_history(self):
        #make a new directory for the coin
        if not os.path.exists("history"):
            os.mkdir("history")
        with open( "history/"+self.coin + "_history.json","w+") as f:
            d = {'coin':self.coin, 'data':self.path, 'history':self.history}
            json.dump(d,f)
            f.flush()
    def run(self):
        self.load_data(self.path)
        self.true_labels()
        self.history = []
        
        combinations = get_combination(8)
        for p in range(1,MAX_PERIOD+1):
            self.X = self.get_features(p*96*2)
            for c in combinations:
                setup = {}
                acc = self.train(self.X,self.Y,p,c)
                setup['period'] = p
                setup['combination']=c
                setup['accuracy'] = acc
                self.history.append(setup)
                self.update_history()
                print(setup)
       
        



if __name__ == '__main__':
    #read the downloaded folder, return a list of all the csv files and call the main function
    files = os.listdir('downloaded')
    for file in files:
        if file.endswith('.csv'):
            coin = file.split('_')[0] + '_'+  file.split('_')[1]
            main = KNN(coin,os.path.join('downloaded',file))
            main.run()
            #after done, rename the file to end with _done
            os.rename(os.path.join('downloaded',file),os.path.join('downloaded',file+'_done'))




    #main = KNN('GRTUSDT','GRTUSDT_15m_data.csv')
    #main.run()
