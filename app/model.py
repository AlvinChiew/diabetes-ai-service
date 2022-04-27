# TBD: add prediction probability 
# TBD: feature selection study to remove unimportant/non-impactful feature

import pickle
import pandas as pd

from typing import Dict

class ObesityPredictor():
    def __init__(self, path_to_weights = '.model/obesity_predictor_gboost.pkl'):
        
        self._FEATURE_NAMES = ['age', 'height', 'weight', 'frequency_of_consumption_of_vegetables', 'number_of_main_meals', 'consumption_of_food_between_meals', 'consumption_of_water_daily', 'physical_activity_frequency', 'time_using_technology_devices', 'consumption_of_alcohol', 'gender_Female', 'gender_Male', 'family_history_with_overweight_no', 'family_history_with_overweight_yes', 'frequent_consumption_of_high_caloric_food_no', 'frequent_consumption_of_high_caloric_food_yes', 'smoke_no', 'smoke_yes', 'calories_consumption_monitoring_no', 'calories_consumption_monitoring_yes', 'transportation_used_Automobile', 'transportation_used_Bike', 'transportation_used_Motorbike', 'transportation_used_Public_Transportation', 'transportation_used_Walking']
        self._INPUT_NAMES = ['gender', 'age', 'height', 'weight', 'family_history_with_overweight', 'frequent_consumption_of_high_caloric_food', 'frequency_of_consumption_of_vegetables', 'number_of_main_meals', 'consumption_of_food_between_meals', 'smoke', 'consumption_of_water_daily', 'calories_consumption_monitoring', 'physical_activity_frequency', 'time_using_technology_devices', 'consumption_of_alcohol', 'transportation_used']
        self._VALID_OPTIONS = {'gender': ['Female', 'Male'], 'family_history_with_overweight': ['yes', 'no'], 'frequent_consumption_of_high_caloric_food': ['no', 'yes'], 'consumption_of_food_between_meals': ['Sometimes',  'Frequently',  'Always',  'no'], 'smoke': ['no', 'yes'], 'calories_consumption_monitoring': ['no', 'yes'], 'consumption_of_alcohol': ['no', 'Sometimes', 'Frequently', 'Always'], 'transportation_used': ['Public_Transportation',  'Walking',  'Automobile',  'Motorbike',  'Bike']}
        self.path_to_weights = path_to_weights        
        with open(self.path_to_weights, 'rb') as f:
            self.classifier = pickle.load(f)

    def predict(self, data: Dict):

        def encode_input(df):
            ordinal_cols = ['consumption_of_food_between_meals', 'consumption_of_alcohol']
            one_hot_cols = ['gender', 'family_history_with_overweight', 'frequent_consumption_of_high_caloric_food', 'smoke', 'calories_consumption_monitoring', 'transportation_used']

            # ordinal 
            enc_freq = {'no': 0, 'Sometimes': 1, 'Frequently': 2, 'Always': 3}
            for ordinal_col in ordinal_cols:
                df[ordinal_col] = df[ordinal_col].map(enc_freq)

            # one-hot
            df_one_hot = pd.get_dummies(df[one_hot_cols])

            df = df.merge(df_one_hot, left_index=True, right_index=True)
            df = df.drop(columns=one_hot_cols)
            return df
        
        def data_to_input(data: Dict) -> str:
            df = pd.DataFrame(data=[data.values()], index=[0], columns=data.keys())
            dict_raw = encode_input(df).squeeze().to_dict()
            dict_input = {}
            for feature in self._FEATURE_NAMES:
                if feature in dict_raw.keys():
                    dict_input[feature] = dict_raw[feature]
                else:
                    dict_input[feature] = 0

            df_input = pd.DataFrame(data=[dict_input.values()], index=[0], columns=dict_input.keys())
            return df_input

        def validate_data(data):
            if data is None or data == {} or data == [] or data == [{}]:
                return "ERROR: missing data"

            if sorted(data.keys()) != sorted(self._INPUT_NAMES):
                return f"ERROR: feature mismatch; expected features: [{','.join(self._INPUT_NAMES)}]"

            for categorical_col in self._VALID_OPTIONS.keys():
                if data[categorical_col] not in self._VALID_OPTIONS[categorical_col]:
                    return f"ERROR: invalid input for `{categorical_col}`;  options: {self._VALID_OPTIONS[categorical_col]}" 

            return "PASS"
        
        data_validity = validate_data(data)
        if data_validity != "PASS":
            return data_validity

        df_input = data_to_input(data)     
        result = self.classifier.predict(df_input)[0]

        return result
