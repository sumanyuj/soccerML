import csv
import pandas as pd
import numpy as np
#from sklearn.ensemble import RandomForestRegressor
#from sklearn.datasets import make_regression
#from sklearn.model_selection import train_test_split
#from sklearn.metrics import mean_squared_error
#use pandas to remove columns and remove data 
currentindex= 0
data = pd.read_csv('inputfile.csv' )
#print(df)#goals_scored, assists, minutes, influence, threat, now_cost
filtered_informaiton = data[[ 'first_name', 'second_name',  'goals_scored', 'assists', 'minutes', 'influence', 'threat', 'now_cost']]
print(filtered_informaiton)
filtered_informaiton.to_csv('clean_data2023-2024.csv', index=False)
#used the various sources down below: https://www.geeksforgeeks.org/python-extracting-rows-using-pandas-iloc/, https://pandas.pydata.org/docs/user_guide/indexing.html, https://www.geeksforgeeks.org/filter-pandas-dataframe-based-on-index/
#https://blog.hubspot.com/website/filter-rows-pandas, https://www.geeksforgeeks.org/python-extracting-rows-using-pandas-iloc/, https://www.geeksforgeeks.org/indexing-and-selecting-data-with-pandas/
#additional resource that I used: https://www.datacamp.com/tutorial/save-as-csv-pandas-dataframe