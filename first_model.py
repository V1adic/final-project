import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

df = pd.read_csv('Data/result_train.csv')

df["Gender"] = df["Gender"].astype('category').cat.codes
df["Physical Activity Level"] = df["Physical Activity Level"].astype('category').cat.codes
df["Smoking Status"] = df["Smoking Status"].astype('category').cat.codes
df["Alcohol Consumption"] = df["Alcohol Consumption"].astype('category').cat.codes
df["Diet"] = df["Diet"].astype('category').cat.codes
df["Chronic Diseases"] = df["Chronic Diseases"].astype('category').cat.codes
df["Medication Use"] = df["Medication Use"].astype('category').cat.codes
df["Family History"] = df["Family History"].astype('category').cat.codes
df["Mental Health Status"] = df["Mental Health Status"].astype('category').cat.codes
df["Sleep Patterns"] = df["Sleep Patterns"].astype('category').cat.codes
df["Education Level"] = df["Education Level"].astype('category').cat.codes
df["Income Level"] = df["Income Level"].astype('category').cat.codes

y = df["Age (years)"]
X = df.drop('Age (years)', axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# создаем модель линейной регрессии
model = LinearRegression()

# обучаем модель на обучающем наборе
model.fit(X_train, y_train)

# прогнозируем целевые значения на тестовом наборе
y_pred = model.predict(X_test)

# вычисляем среднюю квадратичную ошибку (MSE)
mse = mean_squared_error(y_test, y_pred) # Функция потерь

# вычисляем коэффициент детерминации (R2)
r2 = r2_score(y_test, y_pred)

# печатаем результаты
print("Коэффициент детерминации (точность) (R2):", r2)
print("Средняя квадратичная ошибка (MSE):", mse)

# сохраняем обученную модель в файл
filename = 'Models/model_Linear_Regression.sav'
pickle.dump(model, open(filename, 'wb'))