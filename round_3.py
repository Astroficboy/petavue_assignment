#First let's try to extract json into csv
#I am first trying to do this with only 2017 data, so i will take a flight data for 2017, and weather data for 2017.
import json
import csv
import warnings
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
warnings.filterwarnings("ignore")

json_file = "2017-1.json"
output_csv = "output.csv"

#Let's try to define a function to get json data
def json_to_csv(json_file, csv_file):
    with open(json_file, 'r') as json_loader:
        data = json.load(json_loader)
    with open(csv_file, 'w', newline='') as csv_writer:
        writer = csv.writer(csv_writer)
        writer.writerow(["Date", "Time", "Temperature(C)", "Temperature(F)", "FeelsLike(C)", "FeelsLike(F)", "DewPoint(C)", "DewPoint(F)", 
                         "Humidity", "WindSpeed(Kmph)", "WindSpeed(Miles)", "WindGust(Kmph)", "WindGust(Miles)", "WindDirection(Degree)", 
                         "WindDirection(16Point)", "WindChill(C)", "WindChill(F)", "HeatIndex(C)", "HeatIndex(F)", "Pressure", 
                         "Precipitation(mm)", "CloudCover", "Visibility", "WeatherCode", "WeatherDescription", "WeatherIconUrl"])

        for idx in range(len(data['data']['weather'])):
            for hour in data['data']['weather'][idx]['hourly']:
                writer.writerow([
                    data['data']['weather'][idx]['date'],
                    hour['time'],
                    hour['tempC'],
                    hour['tempF'],
                    hour['FeelsLikeC'],
                    hour['FeelsLikeF'],
                    hour['DewPointC'],
                    hour['DewPointF'],
                    hour['humidity'],
                    hour['windspeedKmph'],
                    hour['windspeedMiles'],
                    hour['WindGustKmph'],
                    hour['WindGustMiles'],
                    hour['winddirDegree'],
                    hour['winddir16Point'],
                    hour['WindChillC'],
                    hour['WindChillF'],
                    hour['HeatIndexC'],
                    hour['HeatIndexF'],
                    hour['pressure'],
                    hour['precipMM'],
                    hour['cloudcover'],
                    hour['visibility'],
                    hour['weatherCode'],
                    hour['weatherDesc'][0]['value'],
                    hour['weatherIconUrl'][0]['value']
                ])

json_to_csv(json_file, output_csv)

output_csv = pd.read_csv('output.csv')

print(output_csv.head())

#Let's try to visualize some of the values and try to find out some pattern if we can
sns.lineplot(output_csv['Temperature(C)'])

numeric_columns = output_csv.select_dtypes(include=['int', 'float']).columns

fig, axs = plt.subplots(nrows=len(numeric_columns), figsize=(10, 8))
for i, column in enumerate(numeric_columns):
    sns.scatterplot(output_csv[column], ax=axs[i], cmap='coolwarm')
    axs[i].set_title(column)
    axs[i].set_yticks([])  # Hide y-axis ticks for cleaner visualization

plt.tight_layout()
plt.show()

numeric_data = output_csv.select_dtypes(include=['int', 'float'])
correlation_matrix = numeric_data.corr()
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.show()

'''*Correlated variables are in bold*
We can see that there is a high correlation between **heat index and wind chill** (which seems obvious), 
then there is a correlation between **pressure and visibility** which can be helpful. **Humidity and cloud cover** have some correlation. **Humidity and precipitation** have a correlation.'''

flight_data = pd.read_csv('On_Time_On_Time_Performance_2017_1.csv')
print(flight_data.head())

print(flight_data.info())
print(flight_data.describe())

pd.set_option('display.max_rows', None)
flight_data.isna().sum()
output_csv.isna().sum()

flight_data = flight_data.dropna(axis=1, how='all')
columns_to_keep = ['FlightDate', 'Quarter', 'Year', 'Month', 'DayofMonth', 'DepTime', 'DepDel15', 'CRSDepTime', 'DepDelayMinutes',
                 'Origin', 'Dest', 'ArrTime', 'CRSArrTime', 'ArrDel15', 'ArrDelayMinutes']
flight_data = flight_data[columns_to_keep]

flight_data_filtered_1 = flight_data[(flight_data['Origin'] == 'ATL') | (flight_data['Dest'] == 'ATL')]
flight_data_filtered_1 = flight_data_filtered_1.rename(columns={'FlightDate':'Date'})
final_filter = output_csv[output_csv['Date'].isin(flight_data_filtered_1['Date'])]

merged_csv = pd.merge(flight_data_filtered_1, final_filter, on='Date', how='inner')
merged_csv.to_csv('merged.csv', index=False)

merged_csv = pd.read_csv('merged.csv')
merged_csv.head()

model1 = RandomForestClassifier()
model2 = XGBClassifier()
merged_dtypes_filtered = merged_csv.select_dtypes(include=['int', 'float'])
x, y = merged_dtypes_filtered.drop(columns=['ArrDel15']), merged_dtypes_filtered['ArrDel15']
xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.2, random_state=42)
x_imputer = SimpleImputer(strategy='mean')
y_imputer = SimpleImputer(strategy='most_frequent')

xtrain_clean_data = x_imputer.fit_transform(xtrain)
xtest_clean_data = x_imputer.fit_transform(xtest)
ytrain_clean_data = y_imputer.fit_transform(np.array(ytrain).reshape(-1,1))
ytest_clean_data = y_imputer.fit_transform(np.array(ytest).reshape(-1, 1))

model1.fit(xtrain_clean_data, ytrain_clean_data)
model2.fit(xtrain_clean_data, ytrain_clean_data)

model1_pred = model1.predict(np.array(xtest_clean_data))
model2_pred = model2.predict(np.array(xtest_clean_data))

cnf_mtrx_1 = confusion_matrix(model1_pred, ytest_clean_data)
cnf_mtrx_2 = confusion_matrix(model2_pred, ytest_clean_data)

disp1 = ConfusionMatrixDisplay(confusion_matrix=cnf_mtrx_1)
disp2 = ConfusionMatrixDisplay(confusion_matrix=cnf_mtrx_2)

disp1.plot()
plt.show()

disp2.plot()
plt.show()
