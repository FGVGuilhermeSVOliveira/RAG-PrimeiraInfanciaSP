import streamlit as st
import pandas as pd
import numpy as np

# Configuração da página
st.set_page_config(
    page_title="Meu Primeiro App",
    page_icon="🚀",
    layout="wide",
)

# Barra lateral
st.sidebar.title("Navegação")
pagina = st.sidebar.radio("Ir para:", ["Início", "Dados", "Sobre"])

# Página: Início
if pagina == "Início":
    st.title("🚀 Meu Primeiro App Streamlit")
    st.write("Bem-vindo! Este é um exemplo de site feito em Python com Streamlit.")

    nome = st.text_input("Como você se chama?", "")
    if nome:
        st.success(f"Olá, {nome}! 👋")

    st.subheader("Um contador interativo")
    numero = st.slider("Escolha um número", 0, 100, 25)
    st.write(f"Você escolheu: **{numero}** — o dobro é **{numero * 2}**.")

# Página: Dados
elif pagina == "Dados":
    st.title("📊 Exemplo de Dados")
    st.write("Um gráfico gerado com dados aleatórios.")

    dados = pd.DataFrame(
        np.random.randn(50, 3),
        columns=["Coluna A", "Coluna B", "Coluna C"],
    )

    st.line_chart(dados)
    st.subheader("Tabela")
    st.dataframe(dados, use_container_width=True)

# Página: Sobre
else:
    st.title("ℹ️ Sobre")
    st.write(
        """
        Este app foi criado como exemplo para deploy no **Streamlit Community Cloud**.

        - Feito com Python + Streamlit
        - Hospedado gratuitamente
        - Código aberto no GitHub
        """
    )
    st.info("Edite o arquivo `app.py` para personalizar seu site!")
