# Obesity AI Service
`Obesity Prediction` service hosted on AWS ECS. <br>

# WIP
- [x] Review obesity population related publications and obtain dataset
- [x] Perform feature engineering to clean dataset
- [x] Train an AI model to predict individual obesity
- [x] Craete an automated test bench to validate model input nad output behavior
- [x] Craete an API service
- [ ] Create AWS infra configuration to host service on AWS ECS
- [ ] Create CI/CD pipeline (Github to AWS).
- [ ] Perform grid search to optimiza accuracy via hyperparameter tuning
- [ ] Perform data exploratory analysis on real data in raw dataset via BI tool / Python to study population distribution.
- [ ] Extract and show prediction probability 
- [ ] Conduct feature selection study to remove unimportant/non-impactful feature
- [ ] Create interactive web application

## Algorithm
`Gradient boosting classification` from Scikit Learn (alternative to `XGBoost`) is used for reasons below: <br>
1. yields lower bias error with ensemble of decision trees, i.e. higher accuracy generally.
2. only requires absolute feature values for branching, i.e. feature normalization is not required.

## Dataset
Dataset was collected from [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Estimation+of+obesity+levels+based+on+eating+habits+and+physical+condition+) [1]. Features used in the dataset were described on official publication on [Science Direct](https://www.sciencedirect.com/science/article/pii/S2352340919306985?via%3Dihub#tbl1) [2]. 
77% of the data was generated synthetically using the Weka tool and the `SMOTE` filter, 23% of the data was collected directly from users through a web platform. [2]

| Features                       | Description                               | Input Type                                                                                                                 | Survey Question                                                                                                 | Survey Choices                                            |
|--------------------------------|-------------------------------------------|----------------------------------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------|-----------------------------------------------------------|
| Gender                         | Gender                                    | {Female,Male}                                                                                                              | what is your gender?                                                                                            | {Female,Male}                                             |
| Age                            | Age                                       | numeric                                                                                                                    | what is your age?                                                                                               | year                                                      |
| Height                         | Height                                    | numeric                                                                                                                    | what is your height?                                                                                            | meter                                                     |
| Weight                         | Weight                                    | numeric                                                                                                                    | what is your weight?                                                                                            | kg                                                        |
| family_history_with_overweight | Family with history of overweight         | {yes,no}                                                                                                                   | has a family member suffered or suffers from overweight?                                                        | {yes,no}                                                  |
| FAVC                           | Frequent consumption of high caloric food | numeric                                                                                                                    | do you eat high caloric food frequently?                                                                        | {yes,no}                                                  |
| FCVC                           | Frequency of consumption of vegetables    | numeric                                                                                                                    | do you usually eat vegetables in your meals?                                                                    | {never,sometimes,always}                                  |
| NCP                            | Number of main meals                      | numeric                                                                                                                    | how many meals do you have daily?                                                                               | amount                                                    |
| CAEC                           | Consumption of food between meals         | {no,Sometimes,Frequently,Always}                                                                                           | do you eat any food between meals?                                                                              | {no,Sometimes,Frequently,Always}                          |
| SMOKE                          | Smoke                                     | {yes,no}                                                                                                                   | do you smoke?                                                                                                   | {yes,no}                                                  |
| CH2O                           | Consumption of water daily                | numeric                                                                                                                    | how much water do you drink daily?                                                                              | {<1L,1~2L,>2L}                                            |
| SCC                            | Calories consumption monitoring           | {yes,no}                                                                                                                   | do you monitor the calories you eat daily?                                                                      | {yes,no}                                                  |
| FAF                            | Physical activity frequency               | numeric                                                                                                                    | How often do you have physical activity?                                                                        | {0day,1~2days,2~4days,4~5days}                            |
| TUE                            | Time using technology devices             | numeric                                                                                                                    | how much time do you use technological devices such as cell phone, videogames, television, computer and others? | {0-2hrs,3-5hrs,>5hrs}                                     |
| CALC                           | Consumption of alcohol                    | {no,Sometimes,Frequently,Always}                                                                                           | how often do you drink alcohol?                                                                                 | {I do not drink,Sometimes,Frequently,Always}              |
| MTRANS                         | Transportation used                       | {Automobile,Motorbike,Bike,Public_Transportation,Walking}                                                                  | which transportation do you usually use?                                                                        | {Automobile,Motorbike,Bike,Public_Transportation,Walking} |
| NObeyesdad                     | Obesity Level                             | {Insufficient_Weight,Normal_Weight,Overweight_Level_I,Overweight_Level_II,Obesity_Type_I,Obesity_Type_II,Obesity_Type_III} |                                                                                                                 |                                                           |

## References
```
[1] Palechor, F. M., & de la Hoz Manotas, A. (2019). Dataset for estimation of obesity levels based on eating habits and physical condition in individuals from Colombia, Peru and Mexico. Data in Brief, 104344.<br>
[2] De-La-Hoz-Correa, E., Mendoza Palechor, F., De-La-Hoz-Manotas, A., Morales Ortega, R., & SÃ¡nchez HernÃ¡ndez, A. B. (2019). Obesity level estimation software based on decision trees.
```