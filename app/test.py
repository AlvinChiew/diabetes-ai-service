from model import ObesityPredictor


clf = ObesityPredictor()
data =  {'gender': 'Male', 'age': 39.0, 'height': 1.78, 'weight': 96.0, 'family_history_with_overweight': 'nein', 'frequent_consumption_of_high_caloric_food': 'no', 'frequency_of_consumption_of_vegetables': 2.0, 'number_of_main_meals': 3.0, 'consumption_of_food_between_meals': 'Sometimes', 'smoke': 'no', 'consumption_of_water_daily': 3.0, 'calories_consumption_monitoring': 'no', 'physical_activity_frequency': 1.0, 'time_using_technology_devices': 0.0, 'consumption_of_alcohol': 'Frequently', 'transportation_used': 'Automobile'}

print(clf.predict(data))
