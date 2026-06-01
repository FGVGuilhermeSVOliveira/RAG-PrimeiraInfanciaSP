from sentence_transformers import SentenceTransformer
import streamlit as st
from io import BytesIO
import pandas as pd
import numpy as np
import requests
import psutil
import os

# Configuração da página
st.set_page_config(
    page_title="Meu Primeiro App",
    page_icon="🚀",
    layout="wide",
)

# Nome do arquivo parquet com a base já vetorizada
ARQUIVO_PARQUET = "nurturing_care_PMPI (28-05-2026).pqt"
NOME_MODELO = "paraphrase-multilingual-MiniLM-L12-v2"


# ---------------------------------------------------------------------------
# Funções auxiliares com cache
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner="Carregando base vetorizada...")
def carregar_base(url: str) -> pd.DataFrame:
    """Lê o parquet com a base já vetorizada."""
    #return pd.read_parquet(ARQUIVO_PARQUET)
    if "1drv.ms" in url or "sharepoint.com" in url or "onedrive.live.com" in url:
        # Forçar download direto
        direct_url = url.replace("?e=", "?download=1&e=")
        if "download=1" not in direct_url:
            separator = "&" if "?" in direct_url else "?"
            direct_url = f"{direct_url}{separator}download=1"
    else:
        direct_url = url

    response = requests.get(direct_url, headers={"User-Agent": "Mozilla/5.0"}, allow_redirects=True)
    response.raise_for_status()
    return pd.read_parquet(BytesIO(response.content))


@st.cache_resource(show_spinner="Carregando modelo de embeddings...")
def carregar_modelo() -> SentenceTransformer:
    """Carrega (uma única vez) o modelo de embeddings."""
    return SentenceTransformer(NOME_MODELO)


def buscar(
    df: pd.DataFrame,
    pergunta: str,
    metodo: str = "cosseno",
    threshold: float | None = None,
    top_k: int = 5,
    coluna_embedding: str = "embedding",
) -> pd.DataFrame:
    """Faz busca vetorial no DataFrame já vetorizado."""
    if coluna_embedding not in df.columns:
        raise ValueError(
            f"A coluna '{coluna_embedding}' não existe no DataFrame."
        )
    if df.empty:
        return df.copy()

    modelo = carregar_modelo()
    normalizar = metodo in ("cosseno", "dot")
    vetor_pergunta = modelo.encode(
        [pergunta],
        normalize_embeddings=normalizar,
        convert_to_numpy=True,
    )[0]

    matriz = np.vstack(df[coluna_embedding].to_numpy())

    if metodo == "cosseno":
        norm_base = np.linalg.norm(matriz, axis=1, keepdims=True)
        norm_base[norm_base == 0] = 1e-12
        matriz_norm = matriz / norm_base
        vp = vetor_pergunta / (np.linalg.norm(vetor_pergunta) or 1e-12)
        scores = matriz_norm @ vp
        maior_melhor = True
    elif metodo == "dot":
        scores = matriz @ vetor_pergunta
        maior_melhor = True
    elif metodo == "euclidiana":
        scores = np.linalg.norm(matriz - vetor_pergunta, axis=1)
        maior_melhor = False
    else:
        raise ValueError(f"Método '{metodo}' inválido.")

    resultado = df.copy()
    resultado["score"] = scores

    if threshold is not None:
        if maior_melhor:
            resultado = resultado[resultado["score"] >= threshold]
        else:
            resultado = resultado[resultado["score"] <= threshold]

    resultado = resultado.sort_values("score", ascending=not maior_melhor)
    resultado = resultado.head(top_k)
    return resultado.drop(columns=[coluna_embedding]).reset_index(drop=True)


def get_memoria_mb() -> float:
    """Retorna a memória RSS (RAM real) usada pelo processo atual, em MB.
    Considera tudo que o processo Python carregou: modelo, dados, Streamlit, etc.
    """
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024)


# ---------------------------------------------------------------------------
# Barra lateral — Navegação
# ---------------------------------------------------------------------------
st.sidebar.title("Navegação")
pagina = st.sidebar.radio(
    "Ir para:",
    ["Início", "Dados", "Busca Vetorial", "Sobre"],
)

# ---------------------------------------------------------------------------
# Barra lateral — Monitor de memória (cache)
# ---------------------------------------------------------------------------
st.sidebar.divider()
st.sidebar.markdown("**💾 Uso de memória**")

# Inicializa o valor no session_state
if "memoria_mb" not in st.session_state:
    st.session_state.memoria_mb = get_memoria_mb()

LIMITE_MB = 1024  # 1 GB

col_btn, col_bar = st.sidebar.columns([1, 4])
with col_btn:
    if st.button("🔄", key="btn_atualiza_mem", help="Atualizar uso de memória"):
        st.session_state.memoria_mb = get_memoria_mb()
        st.rerun()

with col_bar:
    mb = st.session_state.memoria_mb
    pct = min(mb / LIMITE_MB, 1.0)
    st.progress(pct, text=f"{mb:,.0f} / {LIMITE_MB} MB")

# ---------------------------------------------------------------------------
# Página: Início
# ---------------------------------------------------------------------------
if pagina == "Início":
    st.title("🚀 Meu Primeiro App Streamlit")
    st.write("Bem-vindo! Este é um exemplo de site feito em Python com Streamlit.")

    nome = st.text_input("Como você se chama?", "")
    if nome:
        st.success(f"Olá, {nome}! 👋")

    st.subheader("Um contador interativo")
    numero = st.slider("Escolha um número", 0, 100, 25)
    st.write(f"Você escolheu: **{numero}** — o dobro é **{numero * 2}**.")

# ---------------------------------------------------------------------------
# Página: Dados
# ---------------------------------------------------------------------------
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

# ---------------------------------------------------------------------------
# Página: Busca Vetorial
# ---------------------------------------------------------------------------
elif pagina == "Busca Vetorial":
    # CSS escopado: torna sticky APENAS a 2ª coluna do bloco que contém
    # um marcador específico (evita afetar as colunas internas do formulário).
    st.markdown(
        """
        <style>
        /* Marcador invisível usado para escopar o seletor */
        .marker-busca-vetorial { display: none; }

        /* Coluna da direita: sticky + tema claro forçado */
        div[data-testid="stHorizontalBlock"]:has(.marker-busca-vetorial)
            > div[data-testid="stColumn"]:nth-of-type(2),
        div[data-testid="stHorizontalBlock"]:has(.marker-busca-vetorial)
            > div[data-testid="column"]:nth-of-type(2) {
            position: sticky;
            top: 1rem;
            align-self: flex-start;
            max-height: calc(100vh - 2rem);
            overflow-y: auto;
            background-color: #ffffff;
            color: #1a1a1a;
            border: 1px solid rgba(128, 128, 128, 0.25);
            border-radius: 6px;
            padding: 1.25rem 1.5rem;
            box-shadow: 0 1px 3px rgba(0, 0, 0, 0.05);
        }

        /* Forçar texto escuro em elementos comuns dentro da coluna direita */
        div[data-testid="stHorizontalBlock"]:has(.marker-busca-vetorial)
            > div[data-testid="stColumn"]:nth-of-type(2)
            :is(p, span, div, h1, h2, h3, h4, h5, h6, li, label, summary, strong, em),
        div[data-testid="stHorizontalBlock"]:has(.marker-busca-vetorial)
            > div[data-testid="column"]:nth-of-type(2)
            :is(p, span, div, h1, h2, h3, h4, h5, h6, li, label, summary, strong, em) {
            color: #1a1a1a;
        }

        /* Links em azul */
        div[data-testid="stHorizontalBlock"]:has(.marker-busca-vetorial)
            > div[data-testid="stColumn"]:nth-of-type(2) a,
        div[data-testid="stHorizontalBlock"]:has(.marker-busca-vetorial)
            > div[data-testid="column"]:nth-of-type(2) a {
            color: #1f6feb;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    col_esquerda, col_direita = st.columns([1, 1], gap="large")

    # ---- COLUNA ESQUERDA: formulário e configurações ----
    with col_esquerda:
        # Insere o marcador para o CSS escopado funcionar
        st.markdown(
            '<div class="marker-busca-vetorial"></div>',
            unsafe_allow_html=True,
        )

        st.title("🔎 Busca Vetorial")
        st.write(
            "Faça uma pergunta e encontre os parágrafos mais relevantes "
            "na base de diários oficiais."
        )

        # Carrega a base
        try:
            df = carregar_base(
                "https://fgvbr-my.sharepoint.com/:u:/g/personal/"
                "guilherme_valentim_fgv_br/IQCNUgevEloURb4Rfw16buLcAS5JrufrNu9iNwQwX0GbEuk?e=W6NcyE"
            )
        except FileNotFoundError:
            st.error(
                f"Arquivo `{ARQUIVO_PARQUET}` não encontrado. "
                "Coloque-o na raiz do projeto (junto com o `app.py`)."
            )
            st.stop()

        st.caption(f"Base carregada: **{len(df):,}** parágrafos.")

        # Formulário de busca
        with st.form("form_busca"):
            pergunta = st.text_input(
                "Pergunta",
                placeholder="Ex: contratação de professores de educação infantil",
            )

            col1, col2, col3 = st.columns(3)
            with col1:
                metodo = st.selectbox(
                    "Método",
                    ["cosseno", "euclidiana", "dot"],
                    help=(
                        "cosseno/dot: maior = mais similar. "
                        "euclidiana: menor = mais similar."
                    ),
                )
            with col2:
                usar_threshold = st.checkbox("Usar threshold", value=True)
                threshold = st.number_input(
                    "Valor do threshold",
                    value=0.3 if metodo != "euclidiana" else 1.0,
                    step=0.05,
                    format="%.2f",
                    disabled=not usar_threshold,
                )
            with col3:
                top_k = st.slider("Top K resultados", 1, 50, 5)

            enviado = st.form_submit_button("Buscar", type="primary")

        # Executa a busca e guarda no session_state para a coluna direita usar
        if enviado:
            if not pergunta.strip():
                st.warning("Digite uma pergunta antes de buscar.")
                st.session_state["resultados_busca"] = None
            else:
                with st.spinner("Buscando..."):
                    resultados = buscar(
                        df=df,
                        pergunta=pergunta,
                        metodo=metodo,
                        threshold=threshold if usar_threshold else None,
                        top_k=top_k,
                    )
                st.session_state["resultados_busca"] = resultados
                st.session_state["pergunta_busca"] = pergunta

        # Tabela completa de resultados (abaixo do formulário)
        resultados_tabela = st.session_state.get("resultados_busca")
        if resultados_tabela is not None and not resultados_tabela.empty:
            st.subheader("📊 Tabela de resultados")
            st.dataframe(resultados_tabela, use_container_width=True)

    # ---- COLUNA DIREITA: resultados (fixada na tela) ----
    with col_direita:
        st.subheader("📋 Resultados")

        resultados = st.session_state.get("resultados_busca")
        pergunta_anterior = st.session_state.get("pergunta_busca", "")

        if resultados is None:
            st.info("Faça uma busca à esquerda para ver os resultados aqui.")
        elif resultados.empty:
            st.info(
                "Nenhum resultado dentro do threshold. "
                "Tente afrouxar o valor ou trocar o método."
            )
        else:
            if pergunta_anterior:
                st.caption(f"🔍 Pergunta: *{pergunta_anterior}*")
            st.success(f"{len(resultados)} resultado(s) encontrado(s).")

            # Cada resultado em um expander
            for i, linha in resultados.iterrows():
                titulo = (
                    f"#{i + 1} — {linha.get('municipio', 'N/A')} "
                    f"(score: {linha['score']:.4f})"
                )
                with st.expander(titulo, expanded=(i == 0)):
                    st.write(linha.get("paragrafo", ""))
                    if "url" in linha and pd.notna(linha["url"]):
                        st.markdown(f"🔗 [Fonte]({linha['url']})")

# ---------------------------------------------------------------------------
# Página: Sobre
# ---------------------------------------------------------------------------
else:
    st.title("ℹ️ Sobre")
    st.write(
        """
        Este app foi criado como exemplo para deploy no **Streamlit Community Cloud**.

        - Feito com Python + Streamlit
        - Busca vetorial com Sentence Transformers
        - Hospedado gratuitamente
        - Código aberto no GitHub
        """
    )
    st.info("Edite o arquivo `app.py` para personalizar seu site!")
