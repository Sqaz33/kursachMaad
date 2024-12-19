import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Загрузка данных
df = pd.read_csv('dataset/Space_Corrected.csv')

# Преобразование столбца 'Cost' в числовой формат (если там могут быть пропуски или ошибки)
df['Cost'] = pd.to_numeric(df['Cost'], errors='coerce')

# Преобразование столбца 'Status Mission' в бинарный формат (Success/Failure)
df['Mission Success'] = df['Status Mission'].apply(lambda x: 1 if x == 'Success' else 0)

print(df.shape)
# Фильтрация компаний, для которых есть данные о запуске (не пустые значения в столбце 'Cost')
df = df.dropna(subset=['Cost'])
print(df.shape)


# Подсчитаем количество запусков для каждой компании
launch_counts = df['Company Name'].value_counts()

# Оставляем только компании с более чем 1 запуском
valid_companies = launch_counts[launch_counts >= 2].index

# Фильтруем DataFrame, чтобы оставить только данные для этих компаний
df = df[df['Company Name'].isin(valid_companies)]

# Уникальные компании после фильтрации
companies = df['Company Name'].unique()

# Определение количества строк и столбцов для подграфиков
n_rows = int(np.ceil(len(companies) / 5))  # 3 графика в одном ряду
n_cols = 5

# Настройка графиков
sns.set(style="whitegrid")
fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
axes = axes.flatten()  # Преобразуем axes в одномерный массив для удобства

# Построение графиков для каждой компании
for i, company in enumerate(companies):
    company_data = df[df['Company Name'] == company]
    
    # Строим график зависимости стоимости от успешности миссии для каждой компании
    sns.scatterplot(data=company_data, x='Cost', y='Mission Success', hue='Mission Success', palette='coolwarm', s=100, legend=None, ax=axes[i])

    axes[i].set_xlabel('')  # Убираем подпись оси X
    axes[i].set_ylabel('')  # Убираем подпись оси Y
    axes[i].set_title('')  # Убираем название графика
    
    # Добавляем название компании на график
    axes[i].text(0.5, 1.05, company, ha='center', va='bottom', fontsize=12, transform=axes[i].transAxes)

# Удаляем лишние подграфики, если количество компаний меньше 3 * n_rows
for j in range(i + 1, len(axes)):
    fig.delaxes(axes[j])

# Автоматическая подгонка оформления
plt.tight_layout()
plt.show()
