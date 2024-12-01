import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('Data/train.csv')

start_count_row = df.shape[0]
print("Количество пропусков:")
print(df.isnull().sum())

# Разделите столбец по символу "/"
df[['Blood Pressure s', 'Blood Pressure d']] = df['Blood Pressure (s/d)'].str.split('/', expand=True)

# Удалить исходный столбец "Blood Pressure (s/d)"
df.drop('Blood Pressure (s/d)', axis=1, inplace=True)

df['Blood Pressure s'] = df['Blood Pressure s'].astype('float64')
df['Blood Pressure d'] = df['Blood Pressure d'].astype('float64')
df["Gender"] = df["Gender"].astype('category')
df["Physical Activity Level"] = df["Physical Activity Level"].astype('category')
df["Smoking Status"] = df["Smoking Status"].astype('category')
df["Alcohol Consumption"] = df["Alcohol Consumption"].astype('category')
df["Diet"] = df["Diet"].astype('category')
df["Chronic Diseases"] = df["Chronic Diseases"].astype('category')
df["Medication Use"] = df["Medication Use"].astype('category')
df["Family History"] = df["Family History"].astype('category')
df["Mental Health Status"] = df["Mental Health Status"].astype('category')
df["Sleep Patterns"] = df["Sleep Patterns"].astype('category')
df["Education Level"] = df["Education Level"].astype('category')
df["Income Level"] = df["Income Level"].astype('category')



# Заполните пропуски средним значением (для численных данных) или самой частой категорией (для категориальных данных)
for col in df:

    if df[col].dtype == "category":
        df[col] = df[col].fillna(df[col].mode()[0])
    
    elif df[col].dtype == "float64":
        df[col] = df[col].fillna(df[col].mean())

# Проверка на выбросы
for col in df.select_dtypes(include='number').columns:
    sns.boxplot(data=df, x=col)
    plt.title(f"Боксплот для столбца {col}")
    plt.show()
print(df)

for col in df:
    if df[col].dtype == "float64":
        # Настройка пороговых значений выбросов (межквартальный размах -- 1,4)
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        df[col] = df[col][(df[col] > (q1 - 1.4 * iqr)) & (df[col] < (q3 + 1.4 * iqr))]

df = df.dropna()

print(f"Выбросы составили -> {((start_count_row / df.shape[0]) * 100) - 100}%")

# Сохраним получившийся датафрейм
df.to_csv('Data/result_train.csv', index=False)