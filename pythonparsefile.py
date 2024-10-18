import csv
import pandas as pd
import numpy as np
#from sklearn.ensemble import RandomForestRegressor
#from sklearn.datasets import make_regression
#from sklearn.model_selection import train_test_split
#from sklearn.metrics import mean_squared_error
#use pandas to remove columns and remove data 
currentindex= 0
def parsing():
    with open('cleaned_strikers_only.csv', mode ='r')as file:
        csvFile = csv.reader(file) #reads the file
        for lines in csvFile: 
            lines.filter(items = [3], axis = 0) #get index 3, index 4, 5, 7,15,22
            print(lines)
        return lines 
    with open('new.csv', 'w', newline='') as file:
        newfile= csv.writer( ) #inputs the onlt filtered information 