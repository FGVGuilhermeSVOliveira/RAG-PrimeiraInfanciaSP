"""
Observatório PMPI — Primeira Infância nos municípios de São Paulo.

Explora os Planos Municipais pela Primeira Infância (PMPI) publicados nos
diários oficiais do estado de SP, com busca semântica (RAG), panorama por
eixos do cuidado integral (nurturing care) e leitura do texto integral
dos planos.

Bases:
  - nurturing_care_PMPI (28-05-2026).pqt — parágrafos vetorizados + scores por eixo
  - PMPI SP (02-06-2026).pqt            — texto integral dos planos (PDF extraído)
"""
from __future__ import annotations

import html as html_lib
import os
import re
import unicodedata
from datetime import datetime
from io import BytesIO
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import psutil
import requests
import streamlit as st
from sentence_transformers import SentenceTransformer

# ---------------------------------------------------------------------------
# Configuração geral
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Observatório PMPI — Primeira Infância SP",
    page_icon="🌱",
    layout="wide",
)

PASTA = Path(__file__).parent
ARQUIVO_BASE = PASTA / "nurturing_care_PMPI (28-05-2026).pqt"
ARQUIVO_PLANOS = PASTA / "PMPI SP (02-06-2026).pqt"
URL_BASE = (
    "https://fgvbr-my.sharepoint.com/:u:/g/personal/"
    "guilherme_valentim_fgv_br/IQCNUgevEloURb4Rfw16buLcAS5JrufrNu9iNwQwX0GbEuk?e=W6NcyE"
)
URL_PLANOS = (
    "https://fgvbr-my.sharepoint.com/:u:/g/personal/"
    "guilherme_valentim_fgv_br/IQDj660hS9S2RKIdh0dNM9BnAUy4M7kouAcwe_N7D9CD2TM?e=lQVVec"
)
NOME_MODELO = "paraphrase-multilingual-MiniLM-L12-v2"
LIMITE_MB = 1024  # limite de RAM do Streamlit Community Cloud

# Eixos do cuidado integral (nurturing care, OMS/UNICEF) + estrutura do plano.
# As chaves correspondem aos sufixos das colunas score_/presente_ e aos
# valores de `eixo_dominante` na base.
EIXOS: dict[str, dict[str, str]] = {
    "saúde": {"rotulo": "Saúde", "emoji": "🩺", "cor": "#e63946"},
    "alimentacao_e_nutricao": {"rotulo": "Alimentação e Nutrição", "emoji": "🍎", "cor": "#f77f00"},
    "educacao": {"rotulo": "Educação", "emoji": "📚", "cor": "#457b9d"},
    "cuidado_parental": {"rotulo": "Cuidado Parental", "emoji": "🤱", "cor": "#2a9d8f"},
    "seguranca_e_protecao": {"rotulo": "Segurança e Proteção", "emoji": "🛡️", "cor": "#6a4c93"},
    "estrutura_do_pmpi": {"rotulo": "Estrutura do PMPI", "emoji": "🏛️", "cor": "#8d99ae"},
}
ROTULO_PARA_EIXO = {f"{m['emoji']} {m['rotulo']}": e for e, m in EIXOS.items()}

PAG_BUSCA = "🔎 Busca Semântica"
PAG_PANORAMA = "📊 Panorama dos Municípios"
PAG_LEITOR = "📄 Leitor de Planos"

STOPWORDS_PT = {
    "de", "da", "do", "das", "dos", "em", "no", "na", "nos", "nas", "para",
    "por", "com", "sem", "sob", "sobre", "que", "qual", "quais", "como",
    "uma", "um", "uns", "umas", "os", "as", "ao", "aos", "pelo", "pela",
    "ser", "tem", "são", "foi", "há", "mais", "menos", "entre", "seu", "sua",
}

# ---------------------------------------------------------------------------
# CSS — cartões de resultado, chips de eixo e "folha de papel" (modo documento)
# ---------------------------------------------------------------------------
PAPER_CSS = """
.folha {
    background: #ffffff;
    color: #1f2328;
    max-width: 860px;
    margin: 0 auto 1rem auto;
    padding: 3rem 3.4rem;
    border-radius: 6px;
    box-shadow: 0 2px 14px rgba(0, 0, 0, 0.28);
    font-family: Georgia, 'Times New Roman', serif;
    font-size: 1.02rem;
    line-height: 1.7;
}
.folha h1 {
    font-size: 1.45rem;
    margin: 0 0 0.4rem 0;
    color: #1f2328;
}
.folha .subtitulo {
    font-family: 'Segoe UI', Arial, sans-serif;
    font-size: 0.82rem;
    color: #57606a;
    border-bottom: 2px solid #e1e4e8;
    padding-bottom: 0.9rem;
    margin-bottom: 1.4rem;
}
.folha .sec { margin: 0 0 1.7rem 0; }
.folha .sec-cab {
    font-family: 'Segoe UI', Arial, sans-serif;
    font-size: 0.85rem;
    font-weight: 600;
    color: #1f2328;
    margin-bottom: 0.35rem;
}
.folha .sec-cab .score-doc {
    font-weight: 400;
    color: #57606a;
    margin-left: 0.5rem;
}
.folha .fonte {
    font-family: 'Segoe UI', Arial, sans-serif;
    font-size: 0.76rem;
    color: #57606a;
    margin-top: 0.3rem;
    word-break: break-all;
}
.folha .fonte a { color: #1f6feb; }
.folha mark { background: #fff3a3; color: #1f2328; padding: 0 2px; border-radius: 2px; }
.folha .texto-doc { white-space: pre-wrap; }
.folha .rodape {
    font-family: 'Segoe UI', Arial, sans-serif;
    font-size: 0.75rem;
    color: #8b949e;
    border-top: 1px solid #e1e4e8;
    padding-top: 0.8rem;
    margin-top: 1.6rem;
    text-align: center;
}
.chip-doc {
    display: inline-block;
    font-family: 'Segoe UI', Arial, sans-serif;
    font-size: 0.72rem;
    font-weight: 600;
    padding: 1px 9px;
    border-radius: 999px;
    border: 1px solid;
    margin-left: 0.5rem;
    vertical-align: middle;
}
"""

APP_CSS = f"""
<style>
{PAPER_CSS}

/* Chips de eixo nos cartões de resultado */
.chip {{
    display: inline-block;
    padding: 2px 10px;
    border-radius: 999px;
    font-size: 0.75rem;
    font-weight: 600;
    border: 1px solid var(--c);
    color: var(--c);
    background: color-mix(in srgb, var(--c) 12%, transparent);
    margin: 0 6px 4px 0;
    white-space: nowrap;
}}
.chip-solid {{
    background: var(--c);
    color: #ffffff;
}}

/* Cabeçalho de cada cartão de resultado */
.res-cab {{
    display: flex;
    align-items: center;
    gap: 0.6rem;
    flex-wrap: wrap;
    margin-bottom: 0.3rem;
}}
.res-rank {{
    font-size: 0.8rem;
    font-weight: 700;
    opacity: 0.55;
}}
.res-mun {{
    font-size: 1.05rem;
    font-weight: 700;
}}
.res-score {{
    font-size: 0.8rem;
    font-variant-numeric: tabular-nums;
    opacity: 0.75;
    margin-left: auto;
}}
.scorebar {{
    height: 6px;
    border-radius: 3px;
    background: rgba(128, 128, 128, 0.18);
    margin-bottom: 0.7rem;
    overflow: hidden;
}}
.scorebar-fill {{
    height: 100%;
    border-radius: 3px;
    background: linear-gradient(90deg, #74c69d, #2d6a4f);
}}
.res-texto {{
    font-size: 0.95rem;
    line-height: 1.6;
    margin-bottom: 0.6rem;
}}
.res-texto mark {{
    background: #ffd54f;
    color: #1a1a1a;
    padding: 0 2px;
    border-radius: 3px;
}}
</style>
"""
st.markdown(APP_CSS, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Carregamento de dados e modelo (com cache)
# ---------------------------------------------------------------------------
def _ler_parquet(arquivo: Path, url: str) -> pd.DataFrame:
    """Lê um parquet: usa o arquivo local se existir, senão baixa da URL.

    Para links do OneDrive/SharePoint, força o download direto (?download=1).
    """
    if arquivo.exists():
        return pd.read_parquet(arquivo)

    if "1drv.ms" in url or "sharepoint.com" in url or "onedrive.live.com" in url:
        direct_url = url.replace("?e=", "?download=1&e=")
        if "download=1" not in direct_url:
            separador = "&" if "?" in direct_url else "?"
            direct_url = f"{direct_url}{separador}download=1"
    else:
        direct_url = url

    resposta = requests.get(
        direct_url, headers={"User-Agent": "Mozilla/5.0"}, allow_redirects=True
    )
    resposta.raise_for_status()
    return pd.read_parquet(BytesIO(resposta.content))


@st.cache_data(show_spinner="Carregando base vetorizada…")
def carregar_base(url: str = URL_BASE) -> pd.DataFrame:
    """Lê o parquet com a base vetorizada (parágrafos + embeddings)."""
    df = _ler_parquet(ARQUIVO_BASE, url)
    # A coluna `erro` é toda nula nessa base — descarta para economizar espaço.
    return df.drop(columns=["erro"], errors="ignore")


@st.cache_data(show_spinner="Carregando planos municipais…")
def carregar_planos(url: str = URL_PLANOS) -> pd.DataFrame | None:
    """Lê o parquet com o texto integral dos PMPIs (local ou via URL)."""
    try:
        df = _ler_parquet(ARQUIVO_PLANOS, url)
    except Exception:  # noqa: BLE001 — sem os planos, o leitor apenas fica indisponível
        return None
    df["municipio"] = df["municipio"].astype(str).str.strip()
    df["tem_texto"] = df["texto_pdf"].fillna("").str.strip().str.len() > 0
    return df.sort_values("municipio").reset_index(drop=True)


@st.cache_resource(show_spinner="Carregando modelo de embeddings…")
def carregar_modelo() -> SentenceTransformer:
    """Carrega (uma única vez) o modelo de embeddings."""
    return SentenceTransformer(NOME_MODELO)


@st.cache_data(show_spinner="Calculando indicadores por município…")
def resumo_municipios() -> pd.DataFrame:
    """Agrega a base por município: nº de parágrafos e % de presença por eixo."""
    df = carregar_base(URL_BASE)
    aggs: dict = {"paragrafos": ("paragrafo", "size"), "url": ("url", "first")}
    for eixo in EIXOS:
        aggs[f"pct_{eixo}"] = (f"presente_{eixo}", "mean")
    resumo = df.groupby("municipio").agg(**aggs).reset_index()
    return resumo.sort_values("paragrafos", ascending=False).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Busca vetorial
# ---------------------------------------------------------------------------
def vetorizar_pergunta(texto: str, metodo: str) -> np.ndarray:
    """Codifica a pergunta com o mesmo modelo usado na base."""
    modelo = carregar_modelo()
    normalizar = metodo in ("cosseno", "dot")
    return modelo.encode(
        [texto], normalize_embeddings=normalizar, convert_to_numpy=True
    )[0]


def buscar(
    df: pd.DataFrame,
    vetor_pergunta: np.ndarray,
    metodo: str = "cosseno",
    threshold: float | None = None,
    top_k: int = 10,
    excluir: tuple[int, int] | None = None,
) -> pd.DataFrame:
    """Busca vetorial no DataFrame já vetorizado.

    `excluir` remove um parágrafo específico (indice_df, id_paragrafo) do
    resultado — usado na busca por parágrafos semelhantes.
    """
    if df.empty:
        return df.drop(columns=["embedding"], errors="ignore").copy()

    matriz = np.vstack(df["embedding"].to_numpy())

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

    if excluir is not None:
        resultado = resultado[
            ~(
                (resultado["indice_df"] == excluir[0])
                & (resultado["id_paragrafo"] == excluir[1])
            )
        ]

    if threshold is not None:
        if maior_melhor:
            resultado = resultado[resultado["score"] >= threshold]
        else:
            resultado = resultado[resultado["score"] <= threshold]

    resultado = resultado.sort_values("score", ascending=not maior_melhor)
    resultado = resultado.head(top_k)
    return resultado.drop(columns=["embedding"]).reset_index(drop=True)


def score_para_pct(score: float, metodo: str) -> float:
    """Converte o score em fração 0–1 para a barrinha visual."""
    if metodo == "euclidiana":
        return max(0.0, min(1.0, 1.0 / (1.0 + float(score))))
    return max(0.0, min(1.0, float(score)))


# ---------------------------------------------------------------------------
# Destaque de termos da pergunta no texto
# ---------------------------------------------------------------------------
_MAPA_ACENTOS = {
    "a": "aáàâãä", "e": "eéèêë", "i": "iíìîï",
    "o": "oóòôõö", "u": "uúùûü", "c": "cç", "n": "nñ",
}


def _regex_insensivel_acentos(termo: str) -> str:
    """Gera um padrão regex que ignora acentuação (educacao ≈ educação)."""
    partes = []
    for ch in termo.lower():
        base = unicodedata.normalize("NFD", ch)[0]
        if base in _MAPA_ACENTOS:
            grupo = _MAPA_ACENTOS[base]
            partes.append(f"[{grupo}{grupo.upper()}]")
        else:
            partes.append(re.escape(ch))
    return "".join(partes)


def extrair_termos(pergunta: str) -> list[str]:
    """Extrai termos relevantes da pergunta (sem stopwords, ≥3 letras)."""
    palavras = re.findall(r"\w+", pergunta.lower(), flags=re.UNICODE)
    termos = [p for p in palavras if len(p) >= 3 and p not in STOPWORDS_PT]
    return sorted(set(termos), key=len, reverse=True)


def destacar(texto: str, termos: list[str]) -> str:
    """Escapa HTML e envolve os termos encontrados em <mark>."""
    if not termos:
        return html_lib.escape(texto)
    padrao = re.compile(
        r"\b(" + "|".join(_regex_insensivel_acentos(t) for t in termos) + r")",
        flags=re.IGNORECASE,
    )
    # Marca no texto cru com sentinelas, escapa e só então vira <mark>
    marcado = padrao.sub(lambda m: f"\x00{m.group(0)}\x01", texto)
    escapado = html_lib.escape(marcado)
    return escapado.replace("\x00", "<mark>").replace("\x01", "</mark>")


def contar_ocorrencias(texto: str, termo: str) -> int:
    if not termo.strip():
        return 0
    return len(re.findall(_regex_insensivel_acentos(termo.strip()), texto, flags=re.IGNORECASE))


# ---------------------------------------------------------------------------
# Helpers de interface
# ---------------------------------------------------------------------------
def fmt_int(n: int) -> str:
    return f"{n:,}".replace(",", ".")


def chip_eixo(eixo: str, solido: bool = False) -> str:
    meta = EIXOS.get(eixo)
    if meta is None:
        return ""
    classe = "chip chip-solid" if solido else "chip"
    return (
        f'<span class="{classe}" style="--c:{meta["cor"]}">'
        f'{meta["emoji"]} {meta["rotulo"]}</span>'
    )


def _agendar_busca(params: dict) -> None:
    """Callback de botões: agenda uma busca para o próximo rerun."""
    st.session_state["busca_pendente"] = dict(params)


def get_memoria_mb() -> float:
    """Memória RSS (RAM real) do processo atual, em MB."""
    return psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)


# ---------------------------------------------------------------------------
# Modo documento (folha estilo Word) + exportação
# ---------------------------------------------------------------------------
def montar_folha_busca(
    titulo: str, resultados: pd.DataFrame, metodo: str, termos: list[str]
) -> str:
    """Monta o HTML da 'folha' com os resultados, para exibição e exportação."""
    agora = datetime.now().strftime("%d/%m/%Y %H:%M")
    partes = [
        '<div class="folha">',
        f"<h1>Relatório de busca — PMPI/SP</h1>",
        f'<div class="subtitulo">Consulta: <b>{html_lib.escape(titulo)}</b> &nbsp;·&nbsp; '
        f"método: {metodo} &nbsp;·&nbsp; {len(resultados)} resultado(s) &nbsp;·&nbsp; "
        f"gerado em {agora}</div>",
    ]
    for i, linha in resultados.iterrows():
        eixo = str(linha.get("eixo_dominante", ""))
        meta = EIXOS.get(eixo)
        chip = (
            f'<span class="chip-doc" style="color:{meta["cor"]}; border-color:{meta["cor"]}">'
            f'{meta["emoji"]} {meta["rotulo"]}</span>'
            if meta
            else ""
        )
        url = linha.get("url")
        fonte = (
            f'<div class="fonte">Fonte: <a href="{html_lib.escape(str(url))}">'
            f"{html_lib.escape(str(url))}</a></div>"
            if pd.notna(url)
            else ""
        )
        partes.append(
            '<div class="sec">'
            f'<div class="sec-cab">{i + 1}. {html_lib.escape(str(linha["municipio"]))}'
            f'<span class="score-doc">score {linha["score"]:.4f}</span>{chip}</div>'
            f'<div class="texto-doc">{destacar(str(linha["paragrafo"]), termos)}</div>'
            f"{fonte}</div>"
        )
    partes.append(
        '<div class="rodape">Observatório PMPI — Primeira Infância SP · FGV · '
        "busca semântica sobre diários oficiais municipais</div></div>"
    )
    return "".join(partes)


def exportar_html(folha_html: str, titulo: str) -> str:
    """Embrulha a folha em um HTML standalone (estilo visualizador de PDF)."""
    return (
        "<!DOCTYPE html><html lang='pt-BR'><head><meta charset='utf-8'>"
        f"<title>{html_lib.escape(titulo)}</title>"
        f"<style>{PAPER_CSS} body{{background:#525659;margin:0;padding:2rem 1rem;}}</style>"
        f"</head><body>{folha_html}</body></html>"
    )


# ---------------------------------------------------------------------------
# Barra lateral — navegação, memória e sobre
# ---------------------------------------------------------------------------
st.sidebar.title("🌱 Observatório PMPI")
st.sidebar.caption("Primeira infância nos municípios de São Paulo")

pagina = st.sidebar.radio(
    "Navegação", [PAG_BUSCA, PAG_PANORAMA, PAG_LEITOR], key="nav"
)

st.sidebar.divider()
_mb = get_memoria_mb()
st.sidebar.progress(
    min(_mb / LIMITE_MB, 1.0),
    text=f"💾 RAM: {_mb:,.0f} / {LIMITE_MB} MB",
)
if st.sidebar.button("🧹 Limpar caches", use_container_width=True):
    st.cache_data.clear()
    st.cache_resource.clear()
    st.rerun()

with st.sidebar.expander("ℹ️ Sobre esta ferramenta"):
    st.markdown(
        """
        Explora os **Planos Municipais pela Primeira Infância (PMPI)** dos
        municípios paulistas a partir dos diários oficiais.

        - **Busca semântica**: embeddings
          `paraphrase-multilingual-MiniLM-L12-v2` (384d) sobre ~47 mil
          parágrafos.
        - **Eixos**: classificação inspirada no marco *Nurturing Care*
          (OMS/UNICEF) + estrutura do plano.
        - **Bases**: parágrafos vetorizados (28/05/2026) e texto integral
          dos planos (02/06/2026).

        Projeto FGV · São Paulo Primeira Infância.
        """
    )


# ===========================================================================
# Página: Busca Semântica
# ===========================================================================
def pagina_busca() -> None:
    st.title("🔎 Busca semântica nos PMPIs")

    try:
        df = carregar_base(URL_BASE)
    except Exception as exc:  # noqa: BLE001 — qualquer falha de carga é fatal aqui
        st.error(f"Não foi possível carregar a base vetorizada: {exc}")
        st.stop()

    st.caption(
        f"**{fmt_int(len(df))}** parágrafos · **{df['municipio'].nunique()}** municípios · "
        f"modelo MiniLM multilíngue (384d). Pergunte em linguagem natural — a busca é por "
        f"significado, não por palavra exata."
    )

    municipios = sorted(df["municipio"].unique())

    # ---- Formulário de busca -------------------------------------------------
    with st.form("form_busca"):
        pergunta = st.text_input(
            "Pergunta ou tema",
            placeholder="Ex.: metas para redução da mortalidade infantil",
        )
        with st.expander("⚙️ Opções avançadas"):
            c1, c2, c3 = st.columns(3)
            with c1:
                metodo = st.selectbox(
                    "Método de similaridade",
                    ["cosseno", "euclidiana", "dot"],
                    help=(
                        "cosseno/dot: maior = mais similar. "
                        "euclidiana: menor = mais similar."
                    ),
                )
            with c2:
                usar_threshold = st.checkbox("Usar threshold", value=True)
                threshold = st.number_input(
                    "Valor do threshold",
                    value=0.30,
                    step=0.05,
                    format="%.2f",
                    help=(
                        "cosseno/dot: valores típicos 0,25–0,50 (mínimo aceito). "
                        "euclidiana: use ~1,0 (máximo aceito)."
                    ),
                )
            with c3:
                top_k = st.slider("Top K resultados", 1, 50, 10)
            c4, c5 = st.columns(2)
            with c4:
                filtro_mun = st.multiselect(
                    "Restringir a municípios", municipios, placeholder="Todos"
                )
            with c5:
                filtro_eixo = st.multiselect(
                    "Restringir a eixos (dominante)",
                    list(ROTULO_PARA_EIXO),
                    placeholder="Todos",
                )
        enviado = st.form_submit_button("🔎 Buscar", type="primary", use_container_width=True)

    # ---- Histórico de buscas -------------------------------------------------
    historico: list[dict] = st.session_state.get("historico_buscas", [])
    if historico:
        with st.expander(f"🕘 Buscas recentes ({len(historico)})"):
            for j, p in enumerate(historico):
                filtros = []
                if p.get("municipios"):
                    filtros.append(f"{len(p['municipios'])} município(s)")
                if p.get("eixos"):
                    filtros.append(f"{len(p['eixos'])} eixo(s)")
                sufixo = f" · {', '.join(filtros)}" if filtros else ""
                st.button(
                    f"“{p['pergunta']}” · {p['metodo']} · top {p['top_k']}{sufixo}",
                    key=f"hist_{j}",
                    on_click=_agendar_busca,
                    args=(p,),
                    use_container_width=True,
                )

    # ---- Monta os parâmetros da busca (formulário ou ação agendada) ----------
    params: dict | None = None
    if enviado:
        if not pergunta.strip():
            st.warning("Digite uma pergunta antes de buscar.")
        else:
            params = {
                "tipo": "texto",
                "pergunta": pergunta.strip(),
                "metodo": metodo,
                "threshold": float(threshold) if usar_threshold else None,
                "top_k": int(top_k),
                "municipios": list(filtro_mun),
                "eixos": [ROTULO_PARA_EIXO[r] for r in filtro_eixo],
            }
    pendente = st.session_state.pop("busca_pendente", None)
    if params is None and pendente is not None:
        params = pendente

    # ---- Executa a busca -----------------------------------------------------
    if params is not None:
        df_alvo = df
        if params.get("municipios"):
            df_alvo = df_alvo[df_alvo["municipio"].isin(params["municipios"])]
        if params.get("eixos"):
            df_alvo = df_alvo[df_alvo["eixo_dominante"].isin(params["eixos"])]

        if df_alvo.empty:
            st.warning("Os filtros escolhidos não deixaram nenhum parágrafo na base.")
        else:
            excluir = None
            with st.spinner("Buscando parágrafos mais relevantes…"):
                if params["tipo"] == "semelhante":
                    ref = df[
                        (df["indice_df"] == params["indice_df"])
                        & (df["id_paragrafo"] == params["id_paragrafo"])
                    ]
                    if ref.empty:
                        st.warning("Parágrafo de referência não encontrado.")
                        vetor = None
                    else:
                        vetor = np.asarray(
                            ref.iloc[0]["embedding"], dtype=np.float32
                        )
                        excluir = (params["indice_df"], params["id_paragrafo"])
                else:
                    vetor = vetorizar_pergunta(params["pergunta"], params["metodo"])

                if vetor is not None:
                    resultados = buscar(
                        df=df_alvo,
                        vetor_pergunta=vetor,
                        metodo=params["metodo"],
                        threshold=params.get("threshold"),
                        top_k=params["top_k"],
                        excluir=excluir,
                    )
                    st.session_state["resultados_busca"] = resultados
                    st.session_state["params_busca"] = params

            # Atualiza o histórico (apenas buscas por texto)
            if params["tipo"] == "texto":
                historico = [
                    p for p in historico if p["pergunta"] != params["pergunta"]
                ]
                historico.insert(0, params)
                st.session_state["historico_buscas"] = historico[:8]

    # ---- Resultados ----------------------------------------------------------
    resultados: pd.DataFrame | None = st.session_state.get("resultados_busca")
    params_res: dict = st.session_state.get("params_busca", {})

    if resultados is None:
        st.info(
            "Faça uma busca acima para começar. Exemplos: *“formação de "
            "professores de creche”*, *“combate à violência doméstica contra "
            "crianças”*, *“aleitamento materno”*."
        )
        return
    if resultados.empty:
        st.warning(
            "Nenhum resultado dentro do threshold. Afrouxe o valor, aumente o "
            "Top K ou troque o método de similaridade."
        )
        return

    metodo_res = params_res.get("metodo", "cosseno")
    if params_res.get("tipo") == "semelhante":
        titulo_busca = params_res.get("descricao", "parágrafos semelhantes")
        termos: list[str] = []
        st.caption(f"🔁 Mostrando parágrafos semelhantes a: *{titulo_busca}*")
    else:
        titulo_busca = params_res.get("pergunta", "")
        termos = extrair_termos(titulo_busca)
        st.caption(f"🔍 Pergunta: *{titulo_busca}*")

    # KPIs da busca
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Resultados", len(resultados))
    k2.metric("Score médio", f"{resultados['score'].mean():.4f}")
    k3.metric("Municípios", resultados["municipio"].nunique())
    eixo_top = resultados["eixo_dominante"].mode()
    if not eixo_top.empty and eixo_top.iloc[0] in EIXOS:
        m = EIXOS[eixo_top.iloc[0]]
        k4.metric("Eixo mais frequente", f"{m['emoji']} {m['rotulo']}")
    else:
        k4.metric("Eixo mais frequente", "—")

    modo = st.radio(
        "Visualização",
        ["🗂️ Cartões", "📄 Documento", "📊 Tabela"],
        horizontal=True,
        key="modo_visualizacao",
    )

    # ---- Modo cartões --------------------------------------------------------
    if modo == "🗂️ Cartões":
        for pos, linha in resultados.iterrows():
            with st.container(border=True):
                eixo_dom = str(linha.get("eixo_dominante", ""))
                pct = score_para_pct(linha["score"], metodo_res)
                st.markdown(
                    f"""
                    <div class="res-cab">
                        <span class="res-rank">#{pos + 1}</span>
                        <span class="res-mun">{html_lib.escape(str(linha["municipio"]))}</span>
                        {chip_eixo(eixo_dom, solido=True)}
                        <span class="res-score">score {linha["score"]:.4f}</span>
                    </div>
                    <div class="scorebar"><div class="scorebar-fill" style="width:{pct * 100:.0f}%"></div></div>
                    <div class="res-texto">{destacar(str(linha["paragrafo"]), termos)}</div>
                    """,
                    unsafe_allow_html=True,
                )
                outros = [
                    chip_eixo(e)
                    for e in EIXOS
                    if e != eixo_dom and linha.get(f"presente_{e}", 0) == 1
                ]
                if outros:
                    st.markdown(
                        "<div>Também aborda: " + "".join(outros) + "</div>",
                        unsafe_allow_html=True,
                    )

                a1, a2, a3 = st.columns([1, 1, 4])
                with a1:
                    with st.popover("📖 Contexto"):
                        vizinhos = df[
                            (df["indice_df"] == linha["indice_df"])
                            & (
                                df["id_paragrafo"].between(
                                    linha["id_paragrafo"] - 2,
                                    linha["id_paragrafo"] + 2,
                                )
                            )
                        ].sort_values("id_paragrafo")
                        st.caption(
                            "Parágrafos vizinhos no mesmo diário oficial "
                            "(o trecho encontrado está destacado):"
                        )
                        for _, v in vizinhos.iterrows():
                            if v["id_paragrafo"] == linha["id_paragrafo"]:
                                st.markdown(
                                    f'<div class="res-texto"><mark>'
                                    f'{html_lib.escape(str(v["paragrafo"]))}'
                                    f"</mark></div>",
                                    unsafe_allow_html=True,
                                )
                            else:
                                st.markdown(
                                    f'<div class="res-texto">'
                                    f'{html_lib.escape(str(v["paragrafo"]))}</div>',
                                    unsafe_allow_html=True,
                                )
                with a2:
                    st.button(
                        "🔁 Semelhantes",
                        key=f"sim_{linha['indice_df']}_{linha['id_paragrafo']}",
                        on_click=_agendar_busca,
                        args=(
                            {
                                "tipo": "semelhante",
                                "indice_df": int(linha["indice_df"]),
                                "id_paragrafo": int(linha["id_paragrafo"]),
                                "descricao": (
                                    f"parágrafo #{pos + 1} de "
                                    f"{linha['municipio']}"
                                ),
                                "metodo": metodo_res,
                                "threshold": params_res.get("threshold"),
                                "top_k": params_res.get("top_k", 10),
                                "municipios": [],
                                "eixos": [],
                            },
                        ),
                        help="Buscar parágrafos parecidos com este em toda a base",
                    )
                with a3:
                    if pd.notna(linha.get("url")):
                        st.markdown(
                            f'🔗 [Diário oficial de origem]({linha["url"]})'
                        )

    # ---- Modo documento (folha estilo Word, refeita) -------------------------
    elif modo == "📄 Documento":
        folha = montar_folha_busca(titulo_busca, resultados, metodo_res, termos)
        st.markdown(folha, unsafe_allow_html=True)
        d1, d2 = st.columns(2)
        with d1:
            st.download_button(
                "⬇️ Baixar relatório (.html)",
                data=exportar_html(folha, f"Busca PMPI — {titulo_busca}"),
                file_name="relatorio_busca_pmpi.html",
                mime="text/html",
                use_container_width=True,
            )
        with d2:
            st.download_button(
                "⬇️ Baixar resultados (.csv)",
                data=resultados.to_csv(index=False).encode("utf-8-sig"),
                file_name="resultados_busca_pmpi.csv",
                mime="text/csv",
                use_container_width=True,
            )

    # ---- Modo tabela ---------------------------------------------------------
    else:
        st.dataframe(
            resultados,
            use_container_width=True,
            column_order=["municipio", "score", "eixo_dominante", "paragrafo", "url"],
            column_config={
                "municipio": st.column_config.TextColumn("Município"),
                "score": st.column_config.NumberColumn("Score", format="%.4f"),
                "eixo_dominante": st.column_config.TextColumn("Eixo dominante"),
                "paragrafo": st.column_config.TextColumn("Parágrafo", width="large"),
                "url": st.column_config.LinkColumn("Fonte", display_text="abrir 🔗"),
            },
        )
        st.download_button(
            "⬇️ Baixar resultados (.csv)",
            data=resultados.to_csv(index=False).encode("utf-8-sig"),
            file_name="resultados_busca_pmpi.csv",
            mime="text/csv",
        )


# ===========================================================================
# Página: Panorama dos Municípios
# ===========================================================================
def pagina_panorama() -> None:
    st.title("📊 Panorama dos municípios")
    st.caption(
        "Como os eixos do cuidado integral (*nurturing care*) aparecem nos "
        "PMPIs publicados pelos municípios paulistas."
    )

    try:
        df = carregar_base(URL_BASE)
        resumo = resumo_municipios()
    except Exception as exc:  # noqa: BLE001
        st.error(f"Não foi possível carregar a base: {exc}")
        st.stop()
    planos = carregar_planos()

    # ---- KPIs ----------------------------------------------------------------
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Municípios na base", df["municipio"].nunique())
    k2.metric("Parágrafos analisados", fmt_int(len(df)))
    if planos is not None:
        k3.metric(
            "Planos com texto integral",
            f"{int(planos['tem_texto'].sum())}/{len(planos)}",
        )
    else:
        k3.metric("Planos com texto integral", "—")
    cols_presente = [f"presente_{e}" for e in EIXOS if e != "estrutura_do_pmpi"]
    pct_tematico = (df[cols_presente].sum(axis=1) > 0).mean()
    k4.metric("Parágrafos com eixo temático", f"{pct_tematico:.0%}")

    st.divider()

    rotulos = [f"{m['emoji']} {m['rotulo']}" for m in EIXOS.values()]
    cores = [m["cor"] for m in EIXOS.values()]
    escala_cores = alt.Scale(domain=rotulos, range=cores)

    c1, c2 = st.columns(2)

    # ---- Distribuição do eixo dominante --------------------------------------
    with c1:
        st.subheader("Eixo dominante dos parágrafos")
        cont = (
            df["eixo_dominante"]
            .value_counts()
            .rename_axis("eixo")
            .reset_index(name="paragrafos")
        )
        cont["rotulo"] = cont["eixo"].map(
            {e: f"{m['emoji']} {m['rotulo']}" for e, m in EIXOS.items()}
        )
        grafico = (
            alt.Chart(cont)
            .mark_bar(cornerRadius=4)
            .encode(
                x=alt.X("paragrafos:Q", title="Parágrafos"),
                y=alt.Y("rotulo:N", sort="-x", title=None),
                color=alt.Color("rotulo:N", scale=escala_cores, legend=None),
                tooltip=[
                    alt.Tooltip("rotulo:N", title="Eixo"),
                    alt.Tooltip("paragrafos:Q", title="Parágrafos", format=","),
                ],
            )
            .properties(height=280)
        )
        st.altair_chart(grafico, use_container_width=True)

    # ---- Cobertura por eixo (% de municípios) --------------------------------
    with c2:
        st.subheader("Municípios que abordam cada eixo")
        cobertura = pd.DataFrame(
            {
                "rotulo": rotulos,
                "pct": [
                    float((resumo[f"pct_{e}"] > 0).mean()) for e in EIXOS
                ],
            }
        )
        grafico2 = (
            alt.Chart(cobertura)
            .mark_bar(cornerRadius=4)
            .encode(
                x=alt.X(
                    "pct:Q",
                    title="% dos municípios",
                    axis=alt.Axis(format=".0%"),
                    scale=alt.Scale(domain=[0, 1]),
                ),
                y=alt.Y("rotulo:N", sort="-x", title=None),
                color=alt.Color("rotulo:N", scale=escala_cores, legend=None),
                tooltip=[
                    alt.Tooltip("rotulo:N", title="Eixo"),
                    alt.Tooltip("pct:Q", title="% municípios", format=".1%"),
                ],
            )
            .properties(height=280)
        )
        st.altair_chart(grafico2, use_container_width=True)

    # ---- Ranking por município -----------------------------------------------
    st.subheader("🏆 Ranking dos municípios")
    st.caption(
        "Percentual de parágrafos do PMPI que mencionam cada eixo. "
        "Clique nos cabeçalhos para reordenar."
    )
    config_colunas: dict = {
        "municipio": st.column_config.TextColumn("Município"),
        "paragrafos": st.column_config.NumberColumn("Parágrafos", format="%d"),
        "url": st.column_config.LinkColumn("Diário", display_text="abrir 🔗"),
    }
    for e, m in EIXOS.items():
        config_colunas[f"pct_{e}"] = st.column_config.ProgressColumn(
            f"{m['emoji']} {m['rotulo']}", format="%.0f%%", min_value=0, max_value=100
        )
    # Percentuais em 0–100 para o formato do ProgressColumn
    tabela_ranking = resumo.copy()
    for e in EIXOS:
        tabela_ranking[f"pct_{e}"] = tabela_ranking[f"pct_{e}"] * 100
    st.dataframe(
        tabela_ranking,
        use_container_width=True,
        height=420,
        column_config=config_colunas,
        column_order=["municipio", "paragrafos"]
        + [f"pct_{e}" for e in EIXOS]
        + ["url"],
        hide_index=True,
    )

    # ---- Raio-X de um município ----------------------------------------------
    st.subheader("🔍 Raio-X de um município")
    municipio = st.selectbox(
        "Escolha um município", resumo["municipio"].sort_values(), key="pan_mun"
    )
    linha_mun = resumo[resumo["municipio"] == municipio].iloc[0]
    df_mun = df[df["municipio"] == municipio]

    r1, r2, r3 = st.columns([1, 1, 2])
    r1.metric("Parágrafos no diário", fmt_int(int(linha_mun["paragrafos"])))
    eixo_moda = df_mun["eixo_dominante"].mode()
    if not eixo_moda.empty and eixo_moda.iloc[0] in EIXOS:
        m = EIXOS[eixo_moda.iloc[0]]
        r2.metric("Eixo mais presente", f"{m['emoji']} {m['rotulo']}")
    with r3:
        if pd.notna(linha_mun["url"]):
            st.markdown(f"🔗 [Diário oficial de {municipio}]({linha_mun['url']})")
        planos_disp = planos[planos["tem_texto"]] if planos is not None else None
        if planos_disp is not None:
            correspondencia = planos_disp[
                planos_disp["municipio"].str.casefold() == str(municipio).casefold()
            ]
            if not correspondencia.empty:
                st.button(
                    "📄 Abrir plano completo no Leitor",
                    on_click=lambda mun=correspondencia.iloc[0]["municipio"]: (
                        st.session_state.update(
                            nav=PAG_LEITOR, leitor_mun=mun, leitor_pag=1
                        )
                    ),
                )

    dados_eixos = pd.DataFrame(
        {
            "rotulo": rotulos,
            "pct": [float(linha_mun[f"pct_{e}"]) for e in EIXOS],
        }
    )
    grafico3 = (
        alt.Chart(dados_eixos)
        .mark_bar(cornerRadius=4)
        .encode(
            x=alt.X(
                "pct:Q",
                title="% dos parágrafos que mencionam o eixo",
                axis=alt.Axis(format=".0%"),
                scale=alt.Scale(domain=[0, 1]),
            ),
            y=alt.Y("rotulo:N", sort="-x", title=None),
            color=alt.Color("rotulo:N", scale=escala_cores, legend=None),
            tooltip=[
                alt.Tooltip("rotulo:N", title="Eixo"),
                alt.Tooltip("pct:Q", title="% parágrafos", format=".1%"),
            ],
        )
        .properties(height=240)
    )
    st.altair_chart(grafico3, use_container_width=True)


# ===========================================================================
# Página: Leitor de Planos
# ===========================================================================
TAMANHO_PAGINA = 4000  # caracteres por "página" do leitor


def paginar_texto(texto: str, tamanho: int = TAMANHO_PAGINA) -> list[str]:
    """Divide o texto em páginas de ~`tamanho` caracteres, sem cortar linhas."""
    paginas: list[str] = []
    atual: list[str] = []
    total = 0
    for linha in texto.split("\n"):
        atual.append(linha)
        total += len(linha) + 1
        if total >= tamanho:
            paginas.append("\n".join(atual))
            atual, total = [], 0
    if atual:
        paginas.append("\n".join(atual))
    return paginas or [""]


def _mudar_pagina(delta: int, total: int) -> None:
    atual = int(st.session_state.get("leitor_pag", 1))
    st.session_state["leitor_pag"] = min(max(atual + delta, 1), total)


def _resetar_pagina() -> None:
    st.session_state["leitor_pag"] = 1


def pagina_leitor() -> None:
    st.title("📄 Leitor de planos municipais")

    planos = carregar_planos()
    if planos is None:
        st.warning(
            f"Não foi possível carregar `{ARQUIVO_PLANOS.name}`. Coloque o arquivo "
            "junto ao `app3.py` ou verifique o link de acesso (`URL_PLANOS`)."
        )
        return

    disponiveis = int(planos["tem_texto"].sum())
    st.caption(
        f"Texto integral extraído dos PDFs dos PMPIs: **{disponiveis}** planos "
        f"disponíveis de **{len(planos)}** municípios coletados."
    )

    e1, e2 = st.columns([2, 2])
    with e1:
        municipio = st.selectbox(
            "Município",
            planos["municipio"],
            key="leitor_mun",
            on_change=_resetar_pagina,
        )
    registro = planos[planos["municipio"] == municipio].iloc[0]

    if not registro["tem_texto"]:
        erro = registro.get("erro")
        st.error(
            "O texto deste plano não pôde ser baixado."
            + (f"\n\n**Detalhe técnico:** `{erro}`" if pd.notna(erro) else "")
        )
        if pd.notna(registro.get("url")):
            st.markdown(f"🔗 [Tentar abrir o documento original]({registro['url']})")
        return

    texto: str = str(registro["texto_pdf"])
    paginas = paginar_texto(texto)
    total_paginas = len(paginas)
    palavras = len(re.findall(r"\w+", texto, flags=re.UNICODE))

    with e2:
        termo = st.text_input(
            "Buscar no documento",
            placeholder="Ex.: creche, conselho tutelar…",
            key="leitor_termo",
        )

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Caracteres", fmt_int(len(texto)))
    m2.metric("Palavras", fmt_int(palavras))
    m3.metric("Páginas no leitor", total_paginas)
    if pd.notna(registro.get("url")):
        with m4:
            st.markdown(f"🔗 [PDF original]({registro['url']})")

    # ---- Busca dentro do documento ------------------------------------------
    termos_doc: list[str] = []
    if termo.strip():
        termos_doc = [termo.strip()]
        ocorrencias_por_pagina = [
            contar_ocorrencias(p, termo) for p in paginas
        ]
        paginas_com_match = [
            i + 1 for i, n in enumerate(ocorrencias_por_pagina) if n > 0
        ]
        total_ocorrencias = sum(ocorrencias_por_pagina)
        if total_ocorrencias == 0:
            st.warning(f"Nenhuma ocorrência de “{termo}” neste plano.")
        else:
            # Termo novo → salta direto para a primeira página com ocorrência
            if st.session_state.get("_leitor_termo_aplicado") != termo:
                st.session_state["_leitor_termo_aplicado"] = termo
                st.session_state["leitor_pag"] = paginas_com_match[0]
            st.success(
                f"**{total_ocorrencias}** ocorrência(s) de “{termo}” em "
                f"**{len(paginas_com_match)}** página(s)."
            )
            opcoes = [
                f"Página {i} — {ocorrencias_por_pagina[i - 1]} ocorrência(s)"
                for i in paginas_com_match
            ]

            def _ir_para_ocorrencia() -> None:
                escolha = st.session_state.get("leitor_sel_ocorrencia")
                if escolha:
                    st.session_state["leitor_pag"] = int(escolha.split()[1])

            st.selectbox(
                "Ir para ocorrência",
                opcoes,
                key="leitor_sel_ocorrencia",
                on_change=_ir_para_ocorrencia,
            )

    # ---- Navegação de páginas -------------------------------------------------
    pag_atual = min(
        max(int(st.session_state.get("leitor_pag", 1)), 1), total_paginas
    )
    st.session_state["leitor_pag"] = pag_atual

    n1, n2, n3 = st.columns([1, 3, 1])
    with n1:
        st.button(
            "⬅️ Anterior",
            disabled=pag_atual <= 1,
            on_click=_mudar_pagina,
            args=(-1, total_paginas),
            use_container_width=True,
        )
    with n2:
        st.markdown(
            f"<div style='text-align:center; padding-top:0.45rem'>"
            f"Página <b>{pag_atual}</b> de <b>{total_paginas}</b></div>",
            unsafe_allow_html=True,
        )
    with n3:
        st.button(
            "Próxima ➡️",
            disabled=pag_atual >= total_paginas,
            on_click=_mudar_pagina,
            args=(1, total_paginas),
            use_container_width=True,
        )

    # ---- A folha (estilo Word) ------------------------------------------------
    conteudo = destacar(paginas[pag_atual - 1], termos_doc)
    st.markdown(
        f"""
        <div class="folha">
            <h1>{html_lib.escape(municipio)}</h1>
            <div class="subtitulo">Plano Municipal pela Primeira Infância ·
                página {pag_atual} de {total_paginas}</div>
            <div class="texto-doc">{conteudo}</div>
            <div class="rodape">Texto extraído automaticamente do PDF oficial ·
                Observatório PMPI — Primeira Infância SP</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.download_button(
        "⬇️ Baixar texto completo (.txt)",
        data=texto.encode("utf-8"),
        file_name=f"PMPI_{municipio}.txt",
        mime="text/plain",
    )


# ---------------------------------------------------------------------------
# Roteamento
# ---------------------------------------------------------------------------
if pagina == PAG_BUSCA:
    pagina_busca()
elif pagina == PAG_PANORAMA:
    pagina_panorama()
else:
    pagina_leitor()
