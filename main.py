

#--// Imports // --
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split

# testing to see if this push works as intended!
# this branch should be protected by pull requests

#Globals
CSV_FILE = "clean_data2023-2024.csv"

def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press ⌘F8 to toggle the breakpoint.

def read_and_split(file):
    #// function for reading and splitting the csv file
    #// returns none

    read_file = pd.read_csv(file) #// Read the file using pandas

    #//( first_name,second_name,goals_scored,assists,minutes,influence,threat,now_cost ) => all file headers
    #//( assists, minutes, influence, threat, now_cost ) => x columns
    #//( target: goals ) => y columns

    #--// Main Program //--
    TEST_SIZE = 0.2
    X = pd.read_csv(file, usecols=["assists", "minutes", "influence", "threat", "now_cost"]) #Use only these columns & assign to X
    Y = pd.read_csv(file, usecols=["goals_scored"]) #Use only these columns & assign to Y

    # print(" --- // DATA // ---")
    # print(f"X DATA: {X}")
    # print(f"Y DATA: {Y}")

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y)
    
    print(f"X_TRAIN: {X_train}, X_TEST: {X_test}, Y_TRAIN: {Y_train}, Y_TEST: {Y_test}")
    return None
if __name__ == '__main__':
    read_and_split(CSV_FILE)

