import streamlit as st
from Pages.principal import principal
from Pages.a_exploratoria import a_exploratoria
from Pages.a_qual_dados import a_qual_dados
from Pages.resultados import resultados

st.set_page_config(layout="wide")

# Menu lateral
st.sidebar.title("Menu")
menu = st.sidebar.selectbox('Selecione uma Página', [
    'Principal', 'Análise exploratória', 'Análise de qualidade de dados', 'Resultado do modelo de regressão linear'])

if menu == 'Principal':
    principal.principal()
elif menu == 'Análise exploratória':
    a_exploratoria.a_exploratoria()
elif menu == 'Análise de qualidade de dados':
    a_qual_dados.a_qual_dados()
elif menu == 'Resultado do modelo de regressão linear':
    resultados.resultados()
