import json
import os
import matplotlib.pyplot as plt


#read all json file in "history" folder
def read_json_file():
    json_files = []
    for root, dirs, files in os.walk("history"):
        for file in files:
            if file.endswith(".json"):
                json_files.append(os.path.join(root, file))
    return json_files

for json_file in read_json_file():
    with open(json_file, "r") as f:
        history = json.load(f)
        print(history['coin'])
        acc = []
        max_value = 0
        max_value_index = 0
        for i, value in enumerate(history['history']):
            if value['accuracy'] > max_value:
                max_value = value['accuracy']
                max_value_index = i
            acc.append(value['accuracy'])
        print("max accuracy: ", max_value)
        print(history['history'][max_value_index])
        #sort and plot the accuracy
        acc.sort( reverse=True)
        #create a plot, set the title and labels, display the point values
        plt.plot(acc)
        title = history['coin'] + " loss," + "with " + str(len(history['history'])) + " features combination"
        plt.title(title)

        plt.show()
        
