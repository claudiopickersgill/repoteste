import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error


def resultados():
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

    st.title('3. Resultado do modelo de regressão linear')
    st.subheader('Variáveis independentes (features): Infant_deaths, Adult_mortality, Alcohol_consumption, Hepatitis_B, Measles, BMI, Polio, Diphtheria e Incidents_HIV. Variável dependente (target): Life_expectancy')
    X = df_novo[['Infant_deaths', 'Adult_mortality', 'Alcohol_consumption',
                 'Hepatitis_B', 'Measles', 'BMI', 'Polio', 'Diphtheria', 'Incidents_HIV',
                 'Thinness_five_nine_years']]
    y = df_novo['Life_expectancy']
    # Dividindo a base de dados em conjuntos de treinamento e teste
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)
    # Criando o modelo de regressão linear
    model = LinearRegression()
    # Treinando o modelo
    model.fit(X_train, y_train)
    # Fazendo previsões no conjunto de teste
    y_pred = model.predict(X_test)
    # Avaliando o desempenho do modelo
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    # Exibir as métricas de avaliação em uma tabela
    metric_data = {'Métrica': ['Mean Squared Error (MSE)', 'Root Mean Squared Error (RMSE)', 'Mean Absolute Error (MAE)'],
                   'Valor': [mse, rmse, mae]}
    metric_df = pd.DataFrame(metric_data)
    st.table(metric_df)

    st.subheader('Resultado da regressão linear')
    # Selecionando algumas observações para exibir no gráfico
    sample_size = 20
    sample_indices = np.random.choice(
        range(len(y_test)), size=sample_size, replace=False)

    # Criando um gráfico de dispersão com as previsões e observações reais
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.scatterplot(x=y_test.iloc[sample_indices], y=y_pred[sample_indices],
                    color='blue', label='Observações Reais vs. Previsões')
    # sns.lineplot(x=float(y_test.iloc[sample_indices]), y=float(y_test.iloc[sample_indices]), color='red', label='Linha de Regressão')
    sns.lineplot(x=y_test.iloc[sample_indices].values,
                 y=y_test.iloc[sample_indices].values, color='red', label='Linha de Regressão')
    ax.set_xlabel('Observações Reais')
    ax.set_ylabel('Previsões')
    ax.set_title('Exemplos da Regressão Linear')
    ax.legend()
    st.pyplot(fig)
