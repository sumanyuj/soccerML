import csv
import pandas as pd
import numpy as np
#from sklearn.ensemble import RandomForestRegressor
#from sklearn.datasets import make_regression
#from sklearn.model_selection import train_test_split
#from sklearn.metrics import mean_squared_error
#use pandas to remove columns and remove data 
def parsing():
    with open('cleaned_strikers_only.csv', mode ='r')as file:
        csvFile = csv.reader(file)
        for lines in csvFile:
            print(lines)
        return lines 