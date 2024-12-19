import pandas as pd
import numpy as np
import statsmodels.api as sm
import seaborn as sns
import matplotlib.pyplot as plt

# Загрузка данных
df = pd.read_csv('dataset/Space_Corrected.csv')

# Преобразование столбца 'Cost' в числовой формат (если там могут быть пропуски или ошибки)
df['Cost'] = pd.to_numeric(df['Cost'], errors='coerce')

# Преобразование столбца 'Status Mission' в бинарный формат (Success/Failure)
df['Mission Success'] = df['Status Mission'].apply(lambda x: 1 if x == 'Success' else 0)

# Фильтрация данных: исключаем строки с пропущенными значениями в столбцах 'Cost' и 'Mission Success'
df = df.dropna(subset=['Cost', 'Mission Success'])

# Подготовка данных для логистической регрессии
X = df['Cost']  # Стоимость миссии
y = df['Mission Success']  # Успех миссии

# Добавляем константный столбец для модели
X = sm.add_constant(X)

# Создаем и обучаем модель логистической регрессии
model = sm.Logit(y, X)
result = model.fit()

# Выводим результаты
print(result.summary())

# Проводим визуализацию зависимости
sns.scatterplot(x='Cost', y='Mission Success', data=df, hue='Mission Success', palette='coolwarm')
plt.xlabel('Cost (in millions)')
plt.ylabel('Mission Success (0=Failure, 1=Success)')
plt.title('Mission Success vs Cost')
plt.show()
