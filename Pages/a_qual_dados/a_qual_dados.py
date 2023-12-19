import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from sklearn.preprocessing import StandardScaler


def a_qual_dados():
    # Carregando a base de dados
    try:
        df = pd.read_csv(
            'https://raw.githubusercontent.com/claudiopickersgill/repoteste/main/data/Life-Expectancy-Data-Updated.csv')
    except Exception as e:
        print(f"Erro ao ler o arquivo CSV: {e}")

    # Reorganizando o índice por ordem alfabética de países e anos
    df_novo = df.sort_values(by=['Country', 'Year']).reset_index(drop=True)

    # Criando uma nova variável de status socioeconômico a partir de duas colunas existentes
    df_novo['Economy_status'] = df_novo.apply(
        lambda row: 1 if row['Economy_status_Developed'] == 1 else 0, axis=1)
    df_novo = df_novo.drop(
        ['Economy_status_Developing', 'Economy_status_Developed'], axis=1)

    st.title('2. Análise da qualidade de dados')
    st.write('Antes da análise da qualidade de dados')
    st.write(df.head(15))
    st.write('Reorganizando o índice por ordem alfabética de países e anos')
    st.write(df_novo.head(15))
    st.write('Criando uma nova variável de status socioeconômico a partir de duas colunas existentes: 0=developing, 1= developed')
    colunas = ['Country', 'Region', 'Year', 'Economy_status']
    st.write(df_novo[colunas])
    st.write('Normalização e escalonamento dos dados')
    lista_colunas = [
        'Infant_deaths', 'Under_five_deaths', 'Adult_mortality', 'Alcohol_consumption', 'Hepatitis_B', 'Measles',
        'BMI', 'Polio', 'Diphtheria', 'Incidents_HIV', 'GDP_per_capita', 'Population_mln', 'Thinness_ten_nineteen_years',
        'Thinness_five_nine_years', 'Schooling', 'Life_expectancy']
    scaler = StandardScaler()
    df_novo[lista_colunas] = scaler.fit_transform(df_novo[lista_colunas])
    st.write(df_novo.head(15))
