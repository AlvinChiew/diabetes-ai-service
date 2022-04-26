# pytest test_model.py
import sys
sys.path.insert(1, '../')  # add project root dir to `PYTHONPATH` for relative import


from app.model import ObesityPredictor


INPUT_NAMES = ['gender', 'age', 'height', 'weight', 'family_history_with_overweight', 'frequent_consumption_of_high_caloric_food', 'frequency_of_consumption_of_vegetables', 'number_of_main_meals', 'consumption_of_food_between_meals', 'smoke', 'consumption_of_water_daily', 'calories_consumption_monitoring', 'physical_activity_frequency', 'time_using_technology_devices', 'consumption_of_alcohol', 'transportation_used']

data_normal = {'gender': 'Female', 'age': 23.0, 'height': 1.63, 'weight': 55.0, 'family_history_with_overweight': 'yes', 'frequent_consumption_of_high_caloric_food': 'no', 'frequency_of_consumption_of_vegetables': 3.0, 'number_of_main_meals': 3.0, 'consumption_of_food_between_meals': 'no', 'smoke': 'no', 'consumption_of_water_daily': 2.0, 'calories_consumption_monitoring': 'yes', 'physical_activity_frequency': 2.0, 'time_using_technology_devices': 1.0, 'consumption_of_alcohol': 'no', 'transportation_used': 'Public_Transportation'}
data_obesity_i = {'gender': 'Male', 'age': 39.0, 'height': 1.78, 'weight': 96.0, 'family_history_with_overweight': 'yes', 'frequent_consumption_of_high_caloric_food': 'no', 'frequency_of_consumption_of_vegetables': 2.0, 'number_of_main_meals': 3.0, 'consumption_of_food_between_meals': 'Sometimes', 'smoke': 'no', 'consumption_of_water_daily': 3.0, 'calories_consumption_monitoring': 'no', 'physical_activity_frequency': 1.0, 'time_using_technology_devices': 0.0, 'consumption_of_alcohol': 'Frequently', 'transportation_used': 'Automobile'}
data_feature_mismatch = {'gender': 'Male', 'ages': 39.0, 'height': 1.78, 'weight': 96.0, 'family_history_with_overweight': 'yes', 'frequent_consumption_of_high_caloric_food': 'no', 'frequency_of_consumption_of_vegetables': 2.0, 'number_of_main_meals': 3.0, 'consumption_of_food_between_meals': 'Sometimes', 'smoke': 'no', 'consumption_of_water_daily': 3.0, 'calories_consumption_monitoring': 'no', 'physical_activity_frequency': 1.0, 'time_using_technology_devices': 0.0, 'consumption_of_alcohol': 'Frequently', 'transportation_used': 'Automobile'}
data_missing_feature = {'genders': 'Male', 'height': 1.78, 'weight': 96.0, 'family_history_with_overweight': 'yes', 'frequent_consumption_of_high_caloric_food': 'no', 'frequency_of_consumption_of_vegetables': 2.0, 'number_of_main_meals': 3.0, 'consumption_of_food_between_meals': 'Sometimes', 'smoke': 'no', 'consumption_of_water_daily': 3.0, 'calories_consumption_monitoring': 'no', 'physical_activity_frequency': 1.0, 'time_using_technology_devices': 0.0, 'consumption_of_alcohol': 'Frequently', 'transportation_used': 'Automobile'}
data_invalid_option = {'gender': 'XYZ', 'age': 23.0, 'height': 1.63, 'weight': 55.0, 'family_history_with_overweight': 'yes', 'frequent_consumption_of_high_caloric_food': 'no', 'frequency_of_consumption_of_vegetables': 3.0, 'number_of_main_meals': 3.0, 'consumption_of_food_between_meals': 'no', 'smoke': 'no', 'consumption_of_water_daily': 2.0, 'calories_consumption_monitoring': 'yes', 'physical_activity_frequency': 2.0, 'time_using_technology_devices': 1.0, 'consumption_of_alcohol': 'no', 'transportation_used': 'Public_Transportation'}

clf = ObesityPredictor(path_to_weights = '../app/.model/obesity_predictor_gboost.pkl')


def test_blank_input():
    assert clf.predict({}) == "ERROR: missing data"


def test_no_input():
    assert clf.predict(None) == "ERROR: missing data"


def test_feature_mismatch_1():
    assert clf.predict(data_feature_mismatch) == f"ERROR: feature mismatch; expected features: [{','.join(INPUT_NAMES)}]"


def test_missing_feature():
    assert clf.predict(data_missing_feature) == f"ERROR: feature mismatch; expected features: [{','.join(INPUT_NAMES)}]"


def test_invalid_option():
    assert clf.predict(data_invalid_option) == f"ERROR: invalid input for `gender`;  options: ['Female', 'Male']" 


def test_normal_input():
    assert clf.predict(data_normal) == "Normal_Weight"


def test_obesity_i_input():
    assert clf.predict(data_obesity_i) == "Obesity_Type_I"
