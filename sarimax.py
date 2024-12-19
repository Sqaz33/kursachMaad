import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
import matplotlib.pyplot as plt

# Загрузка данных
df = pd.read_csv('dataset/Space_Corrected.csv')

# Преобразование даты в формат datetime
df['Datum'] = pd.to_datetime(df['Datum'], format='%a %b %d, %Y %H:%M %Z', errors='coerce')

# Преобразование столбца 'Status Mission' в бинарный формат (Success/Failure)
df['Mission Success'] = df['Status Mission'].apply(lambda x: 1 if x == 'Success' else 0)

# Фильтрация данных с некорректными датами
df = df.dropna(subset=['Datum'])

# Выделение 5 компаний с наибольшим количеством миссий
top_companies = df['Company Name'].value_counts().head(5).index

# Построение графиков для каждой компании
for company in top_companies:
    company_data = df[df['Company Name'] == company].copy()
    company_data['Year'] = company_data['Datum'].dt.year

    # Агрегация данных: успешные и неуспешные миссии по годам
    success_per_year = company_data.groupby('Year')['Mission Success'].sum()
    total_per_year = company_data.groupby('Year')['Mission Success'].count()
    failure_per_year = total_per_year - success_per_year

    # Установка временного индекса
    success_per_year.index = pd.DatetimeIndex(success_per_year.index.astype(str) + '-01-01')
    failure_per_year.index = pd.DatetimeIndex(failure_per_year.index.astype(str) + '-01-01')

    # SARIMAX для успешных миссий
    forecast_success_values = pd.Series(dtype=float)
    forecast_success_index = pd.date_range(start=success_per_year.index[-1], periods=11, freq='Y')[1:]  # Диапазон прогноза
    if len(success_per_year) > 1:  # SARIMAX требует больше 1 точки данных
        model_success = SARIMAX(success_per_year, order=(1, 1, 1), seasonal_order=(0, 1, 1, 12))
        results_success = model_success.fit(disp=False)
        forecast_success = results_success.get_forecast(steps=10)
        forecast_success_values = pd.Series(forecast_success.predicted_mean.values, index=forecast_success_index)

    # SARIMAX для неуспешных миссий
    forecast_failure_values = pd.Series(dtype=float)
    forecast_failure_index = pd.date_range(start=failure_per_year.index[-1], periods=11, freq='Y')[1:]  # Диапазон прогноза
    if len(failure_per_year) > 1:
        model_failure = SARIMAX(failure_per_year, order=(1, 1, 1), seasonal_order=(0, 1, 1, 12))
        results_failure = model_failure.fit(disp=False)
        forecast_failure = results_failure.get_forecast(steps=10)
        forecast_failure_values = pd.Series(forecast_failure.predicted_mean.values, index=forecast_failure_index)

    # График
    plt.figure(figsize=(12, 6))
    plt.plot(success_per_year.index.year, success_per_year, label='Успешные миссии (исторические)', color='blue')
    if not forecast_success_values.empty:
        plt.plot(forecast_success_values.index.year, forecast_success_values, label='Прогноз успешных миссий', linestyle='--', color='blue')

    plt.plot(failure_per_year.index.year, failure_per_year, label='Неуспешные миссии (исторические)', color='red')
    if not forecast_failure_values.empty:
        plt.plot(forecast_failure_values.index.year, forecast_failure_values, label='Прогноз неуспешных миссий', linestyle='--', color='red')

    plt.title(f'Прогноз миссий для компании {company}')
    plt.xlabel('Год')
    plt.ylabel('Количество миссий')
    plt.legend()
    plt.grid(True)
    plt.show()
