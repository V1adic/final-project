import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import tensorflow as tf
import pickle


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

# Разделение данных на обучающий и тестовый набор
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Создание модели глубокой нейронной сети
model = tf.keras.models.Sequential()

# Добавление слоев в нейронную сеть
model.add(tf.keras.layers.Dense(units=10, activation='relu', input_dim=X_train.shape[1]))
model.add(tf.keras.layers.Dense(units=10, activation='relu'))
model.add(tf.keras.layers.Dense(units=1))

# Компиляция модели
model.compile(optimizer='adam', loss='mean_absolute_error', metrics=['mean_squared_error'])

# Обучение модели
model.fit(X_train, y_train, epochs=200, batch_size=32)

# Оценка модели на тестовом наборе
y_pred = model.predict(X_test)

# вычисляем коэффициент детерминации (R2)
r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
print('Среднеквадратичная ошибка (MSE):', mse)
print("Коэффициент детерминации (точность) (R2):", r2)

# сохраняем обученную модель в файл
filename = 'Models/model_Sequential.sav'
pickle.dump(model, open(filename, 'wb'))