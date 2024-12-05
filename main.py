

#--// Imports // --
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score


# testing to see if this push works as intended!
# this branch should be protected by pull requests

#Globals
TRAIN_CSV_FILE = "clean_data2023-2024.csv"
PREDICT_CSV_FILE = "clean_data2022-2023.csv"

def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press ⌘F8 to toggle the breakpoint.
    
class GoalPredictor:
    
    def __init__(self, file, random_state=42):
        self.file = file
        self.random_state = random_state
        self.model = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.feature_columns = ["assists", "minutes", "influence", "threat", "now_cost"]
    
    
    def read_and_split(self,file):
        
        print("SPLITTING THE DATA FILE ....")
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
        
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = Y_train
        self.y_test = Y_test
        
        print(f"X_TRAIN: {X_train}, X_TEST: {X_test}, Y_TRAIN: {Y_train}, Y_TEST: {Y_test}")
        return None


    def train_model(self, n_estimators=100, max_depth=None, min_samples_split=2):
        """
        Trains the Random Forest model w/ the params specified
        return a metrics dictionary to test and evaluate it
        """
        
        print("TRAINING THE MODEL ....")

    
        # This initializes the model from scikit learns 
        self.model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            random_state=self.random_state
        )
        
        # This is the training stage 
        self.model.fit(self.X_train, self.y_train.values.ravel())
        
        # Make predictions
        train_predictions = self.model.predict(self.X_train)
        test_predictions = self.model.predict(self.X_test)
        
        # get metrics by 
        metrics = {
            'train_rmse': np.sqrt(mean_squared_error(self.y_train, train_predictions)),
            'test_rmse': np.sqrt(mean_squared_error(self.y_test, test_predictions)),
            'train_r2': r2_score(self.y_train, train_predictions),
            'test_r2': r2_score(self.y_test, test_predictions)
        }
        
        # Print metrics
        # R^2 tells how well model captures the pattern in goal scoring
        # RMSE (root mean square) tells of the deviation of predictions in terms of goals
        # 
        print("\nModel Performance Metrics:")
        print(f"TRAIN R^2 SCORE: {metrics['train_r2']}")
        print(f"TESTING R^2 Score: {metrics['test_r2']}")
        print(f"Training RMSE: {metrics['train_rmse']}")
        print(f"Testing RMSE: {metrics['test_rmse']}")
        
        # FEATURE IMPORTANCE
        # this lists out the "WEIGHTS" that the model made during training 
        # greater weight = greater importance
        feature_importance = pd.DataFrame({
            'feature': self.feature_columns,
            'importance': self.model.feature_importances_
        })
        feature_importance = feature_importance.sort_values('importance', ascending=False)
        
        print("\nFeature Importance:")
        print(feature_importance)
        
        return metrics
            
      
        
    def predict(self, X):
        print("FORMING PREDICTIONS ....")

        
        """
        Make predictions using the trained model.
        """
        # MAKE THE PREDICTION!!!
        predictions = self.model.predict(X)
        return predictions
                

     
if __name__ == '__main__':
    
    predictor = GoalPredictor(TRAIN_CSV_FILE)
    
    predictor.read_and_split("clean_data2023-2024.csv")

    predictor.train_model(n_estimators=100,max_depth=10)
    
    # prediction now!
    predict_csv_data = pd.read_csv(PREDICT_CSV_FILE, usecols=predictor.feature_columns)
    
    predictions = predictor.predict(predict_csv_data)

    
    player_names = pd.read_csv(PREDICT_CSV_FILE)[['first_name', 'second_name']]
    actual_goals = pd.read_csv(PREDICT_CSV_FILE)[['goals_scored']]

    # GO OVER DATA AND COMPARE ACTUAL TO EXPECTED RESULTS
    results = pd.DataFrame({
        'first_name': player_names['first_name'],
        'second_name': player_names['second_name'],
        'predicted_goals': predictions.round(1),
        'actual goals': actual_goals['goals_scored']
    })
 
    print("\nPREDICTED GOALS")
    print(results)