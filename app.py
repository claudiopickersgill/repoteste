import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler

# Carregando a base de dados
df = pd.read_csv('data\Life-Expectancy-Data-Updated.csv')

# Reorganizando o índice por ordem alfabética de países e anos
df_novo = df.sort_values(by=['Country', 'Year']).reset_index(drop=True)

# Criando uma nova variável de status socioeconômico a partir de duas colunas existentes
df_novo['Economy_status'] = df_novo.apply(lambda row: 1 if row['Economy_status_Developed'] == 1 else 0, axis=1)
df_novo = df_novo.drop(['Economy_status_Developing', 'Economy_status_Developed'], axis=1)

# Criando as funções para definir o conteúdo de cada página
def main():
    st.title('Dashboard de Análise de Expectativa de Vida')
    st.write('Navegue a partir do menu lateral para exibir as diferentes fases do projeto')

def page1():
    st.title('1. Análise exploratória')
    st.subheader('Barplot da expectativa de vida por região')
    fig, ax = plt.subplots()
    sns.barplot(x='Region', y='Life_expectancy', data=df_novo)
    plt.title('Expectativa de vida por região')
    plt.xticks(rotation=90)
    plt.figure(figsize=(15, 8))
    st.pyplot(fig)
    st.subheader('Barplot da mortalidade infantil por região')
    fig, ax = plt.subplots()
    sns.barplot(x='Region', y='Infant_deaths', data=df_novo)
    plt.title('Mortalidade infantil por região')
    plt.xticks(rotation=90)
    plt.figure(figsize=(15, 8))
    st.pyplot(fig)
    st.subheader('Barplot da expectativa de vida por região, dividida por status econômico')
    media_expectativa_status = df_novo.groupby(['Region', 'Year', 'Economy_status'])['Life_expectancy'].mean()
    media_expectativa_status = media_expectativa_status.reset_index()
    fig, ax = plt.subplots()
    sns.barplot(x='Region', y='Life_expectancy', hue='Economy_status', data=media_expectativa_status)
    plt.title('Média da expectativa de vida por região, divididas por status econômico')
    plt.xticks(rotation=90)
    plt.figure(figsize=(15, 8))
    st.pyplot(fig)

def page2():
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

# INSERIR O PLOT DA REGRESSÃO LINEAR
def page3():
    st.title('3. Resultado do modelo de regressão linear')
    st.subheader('Variáveis independentes (features): Infant_deaths, Adult_mortality, Alcohol_consumption, Hepatitis_B, Measles, BMI, Polio, Diphtheria e Incidents_HIV. Variável dependente (target): Life_expectancy')
    X = df_novo[['Infant_deaths', 'Adult_mortality', 'Alcohol_consumption', 
    'Hepatitis_B', 'Measles', 'BMI', 'Polio', 'Diphtheria', 'Incidents_HIV', 
    'Thinness_five_nine_years']]
    y = df_novo['Life_expectancy']
    # Dividindo a base de dados em conjuntos de treinamento e teste
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
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
    sample_indices = np.random.choice(range(len(y_test)), size=sample_size, replace=False)

    # Criando um gráfico de dispersão com as previsões e observações reais
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.scatterplot(x=y_test.iloc[sample_indices], y=y_pred[sample_indices], color='blue', label='Observações Reais vs. Previsões')
    #sns.lineplot(x=float(y_test.iloc[sample_indices]), y=float(y_test.iloc[sample_indices]), color='red', label='Linha de Regressão')
    sns.lineplot(x=y_test.iloc[sample_indices].values, y=y_test.iloc[sample_indices].values, color='red', label='Linha de Regressão')
    ax.set_xlabel('Observações Reais')
    ax.set_ylabel('Previsões')
    ax.set_title('Exemplos da Regressão Linear')
    ax.legend()
    st.pyplot(fig)


# Menu lateral
menu = st.sidebar.selectbox('Selecione uma Página', [
    'Principal', 'Análise exploratória', 'Análise de qualidade de dados', 'Resultado do modelo de regressão linear'])

if menu == 'Principal':
    main()
elif menu == 'Análise exploratória':
    page1()
elif menu == 'Análise de qualidade de dados':
    page2()
elif menu == 'Resultado do modelo de regressão linear':
    page3()