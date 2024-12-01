import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
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

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Создание и обучение модели дерева решений
model = DecisionTreeClassifier()
model.fit(X, y)

# Прогнозирование значений целевой переменной
y_pred = model.predict(X_test)

# Вычисление среднеквадратичной ошибки
mse = mean_squared_error(y_test, y_pred)

# вычисляем коэффициент детерминации (R2)
r2 = r2_score(y_test, y_pred)

# печатаем результаты
print("Коэффициент детерминации (точность) (R2):", r2)
print("Среднеквадратичная ошибка:", mse)

# сохраняем обученную модель в файл
filename = 'Models/model_Decision_Tree_Classifier.sav'
pickle.dump(model, open(filename, 'wb'))
