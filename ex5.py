import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency

# Загрузка данных
file_path = "dataset/space_missions.csv"
data = pd.read_csv(file_path)

# Предобработка данных
data['MissionStatus'] = data['MissionStatus'].str.strip()
success_rate = data.groupby('Company')['MissionStatus'].value_counts(normalize=True).unstack()
mission_counts = data.groupby('Company')['MissionStatus'].value_counts().unstack()

# Хи-квадрат тест
contingency_table = data.pivot_table(index='Company', columns='MissionStatus', aggfunc='size', fill_value=0)
chi2, p, _, _ = chi2_contingency(contingency_table)

# Streamlit дашборд
st.title("Анализ гипотезы: Влияние компании на успех миссий")
st.markdown("Гипотеза: Частота успешных запусков зависит от компании.")

# Визуализация доли успехов
st.subheader("Доля успешных миссий для каждой компании")
fig, ax = plt.subplots(figsize=(12, 6))
if 'Success' in success_rate.columns:
    success_rate['Success'].plot(kind='bar', color='green', ax=ax)
    ax.set_title('Доля успешных миссий для каждой компании')
    ax.set_xlabel('Компания')
    ax.set_ylabel('Доля успехов')
    st.pyplot(fig)
else:
    st.warning("Нет данных о статусе 'Success' для построения графика.")

# Таблица с количеством миссий
st.subheader("Частоты успешных и неуспешных миссий")
st.table(mission_counts)

# Вывод результатов хи-квадрат теста
st.subheader("Результаты статистического теста")
st.markdown(f"**Хи-квадрат значение**: {chi2:.2f}")
st.markdown(f"**p-значение**: {p:.4f}")
if p < 0.05:
    st.markdown("**Вывод:** Зависимость между компанией и успехом миссий статистически значима.")
else:
    st.markdown("**Вывод:** Зависимости между компанией и успехом миссий не выявлено.")

# Дополнительный график: Успехи внутри компании
selected_company = st.selectbox("Выберите компанию для анализа:", data['Company'].unique())
company_data = data[data['Company'] == selected_company]
company_status_counts = company_data['MissionStatus'].value_counts(normalize=True)

st.subheader(f"Успехи миссий компании {selected_company}")
fig, ax = plt.subplots(figsize=(6, 6))
company_status_counts.plot(kind='pie', autopct='%1.1f%%', startangle=90, ax=ax)
ax.set_ylabel('')
ax.set_title(f'Распределение статуса миссий: {selected_company}')
st.pyplot(fig)
