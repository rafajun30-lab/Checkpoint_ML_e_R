# -*- coding: utf-8 -*-
"""
Checkpoint - Machine Learning & Modelling + Statistical Computing with R & Python
Dataset: WINES (Kaggle - wine-dataset-for-clustering)

Integrantes:
    Guilherme Orugian      - RM572882
    Lucas Henrique         - RM571901
    Rafael Jun Aita Hirata - RM569708
    Rodrigo Bettio         - RM573725
    Vinicius Eddo          - RM571008

Como executar:
    1) Coloque o arquivo wines.csv (ou wine-clustering.csv) na mesma pasta deste script.
    2) pip install pandas numpy matplotlib scipy scikit-learn
    3) python checkpoint_wines.py
    Os graficos sao salvos na pasta "graficos".
    Se o CSV nao for encontrado, o script usa o dataset Wine do kaggle
    (https://www.kaggle.com/datasets/harrywang/wine-dataset-for-clustering).
"""

import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA

RANDOM_STATE = 42
PASTA_GRAFICOS = "graficos"
os.makedirs(PASTA_GRAFICOS, exist_ok=True)


def salvar(nome):
    """Salva a figura atual em arquivo e exibe na tela."""
    plt.tight_layout()
    plt.savefig(os.path.join(PASTA_GRAFICOS, nome), dpi=150)
    plt.show()
    plt.close()


def titulo(texto):
    print("\n" + "=" * 70)
    print(texto)
    print("=" * 70)


# ----------------------------------------------------------------------
# 0. CARGA DOS DADOS
# ----------------------------------------------------------------------
def carregar_dados():
    """Le o CSV de vinhos e padroniza os nomes das colunas."""
    candidatos = ["wines.csv", "wine-clustering.csv", "wine_clustering.csv"]
    df = None
    for arq in candidatos:
        if os.path.exists(arq):
            df = pd.read_csv(arq)
            print(f"Arquivo carregado: {arq}")
            break
    if df is None:
        from sklearn.datasets import load_wine
        df = load_wine(as_frame=True).data
        print("CSV nao encontrado -> usando sklearn.datasets.load_wine()")

    # Padroniza nomes: minusculas, sem espacos/simbolos
    novos = []
    for c in df.columns:
        c = re.sub(r"[^0-9a-zA-Z]+", "_", c.strip().lower()).strip("_")
        if "alcalinity" in c or "alcanity" in c:
            c = "ash_alcalinity"
        if c.startswith("od280"):
            c = "od280"
        novos.append(c)
    df.columns = novos

    # Remove colunas de classe/rotulo, se existirem (clustering nao usa rotulo)
    for col in ["class", "target", "customer_segment", "cultivar"]:
        if col in df.columns:
            df = df.drop(columns=col)
    return df


df = carregar_dados()
titulo("VISAO GERAL DO DATASET")
print("Dimensao (linhas, colunas):", df.shape)
print(df.head())
print("\nTipos de dados:")
print(df.dtypes)
print("\nValores nulos por coluna:")
print(df.isnull().sum())




# ======================================================================
# PARTE 1 - STATISTICAL COMPUTING (Python)
# Variaveis quantitativas escolhidas:
#   - alcohol    (teor alcoolico)
#   - malic_acid (acido malico)
# ======================================================================



VARIAVEIS = ["alcohol", "malic_acid"]

# ----------------------------------------------------------------------
# 1a) VISUALIZACAO DOS DADOS: tabela de distribuicao de frequencias + grafico
# ----------------------------------------------------------------------
titulo("PARTE 1a - TABELA DE DISTRIBUICAO DE FREQUENCIAS E GRAFICOS")


def tabela_frequencias(serie):
    """Constroi a tabela de frequencias usando a regra de Sturges
    para definir o numero de classes: k = 1 + 3,322 * log10(n)."""
    n = len(serie)
    k = int(np.ceil(1 + 3.322 * np.log10(n)))
    classes = pd.cut(serie, bins=k, include_lowest=True)
    fi = classes.value_counts().sort_index()
    tabela = pd.DataFrame({
        "Classe": fi.index.astype(str),
        "fi": fi.values,
    })
    tabela["fr"] = tabela["fi"] / n                 # frequencia relativa
    tabela["fr%"] = (tabela["fr"] * 100).round(2)   # frequencia relativa (%)
    tabela["Fi"] = tabela["fi"].cumsum()            # frequencia acumulada
    tabela["Fr%"] = tabela["fr%"].cumsum().round(2)  # acumulada relativa (%)
    return tabela.drop(columns="fr"), k


for var in VARIAVEIS:
    tab, k = tabela_frequencias(df[var])
    print(f"\nTabela de frequencias - {var} (k = {k} classes, regra de Sturges)")
    print(tab.to_string(index=False))

# Histograma + poligono de frequencias e boxplot lado a lado
fig, eixos = plt.subplots(2, 2, figsize=(12, 8))
for i, var in enumerate(VARIAVEIS):
    _, k = tabela_frequencias(df[var])
    eixos[0, i].hist(df[var], bins=k, edgecolor="black", alpha=0.75)
    eixos[0, i].set_title(f"Histograma - {var}")
    eixos[0, i].set_xlabel(var)
    eixos[0, i].set_ylabel("Frequencia")
    eixos[1, i].boxplot(df[var], vert=False)
    eixos[1, i].set_title(f"Boxplot - {var}")
    eixos[1, i].set_xlabel(var)
salvar("parte1a_histogramas_boxplots.png")

# Interpretacao (comentarios):
# - alcohol: distribuicao aproximadamente simetrica, concentrada entre ~12 e ~14
#   graus de alcool, sem outliers.
# - malic_acid: distribuicao assimetrica a direita (cauda longa para valores
#   altos) e com 3 outliers acima de ~5,3; a maioria dos vinhos tem pouco
#   acido malico (metade deles abaixo de ~1,9).

# ----------------------------------------------------------------------
# 1b) ANALISE DESCRITIVA: tendencia central, dispersao e separatrizes
# ----------------------------------------------------------------------

titulo("PARTE 1b - ANALISE DESCRITIVA")

resumo = {}
for var in VARIAVEIS:
    x = df[var]
    q1, q2, q3 = x.quantile([0.25, 0.50, 0.75])
    moda = x.mode()
    resumo[var] = {
        # Tendencia central
        "media": x.mean(),
        "mediana": x.median(),
        "moda": moda.iloc[0] if len(moda) else np.nan,
        # Dispersao
        "amplitude": x.max() - x.min(),
        "variancia": x.var(ddof=1),
        "desvio_padrao": x.std(ddof=1),
        "coef_variacao_%": x.std(ddof=1) / x.mean() * 100,
        # Separatrizes (quartis)
        "minimo": x.min(),
        "Q1": q1,
        "Q2 (mediana)": q2,
        "Q3": q3,
        "maximo": x.max(),
        "IQR (Q3-Q1)": q3 - q1,
        # Forma
        "assimetria": stats.skew(x),
        "curtose": stats.kurtosis(x),
    }
resumo_df = pd.DataFrame(resumo).round(4)
print(resumo_df)

# Deteccao de outliers pelo criterio de Tukey (1,5 * IQR)
print("\nOutliers (criterio 1,5*IQR):")
for var in VARIAVEIS:
    q1, q3 = resumo[var]["Q1"], resumo[var]["Q3"]
    iqr = q3 - q1
    inf, sup = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    out = df[(df[var] < inf) | (df[var] > sup)][var]
    print(f"  {var}: limites [{inf:.2f}; {sup:.2f}] -> {len(out)} outlier(s)")

# Interpretacao (comentarios):
# - Se media > mediana (malic_acid), a distribuicao tem assimetria positiva.
# - Se media ~ mediana (alcohol), a distribuicao e aproximadamente simetrica.
# - O coeficiente de variacao (CV) compara a dispersao relativa das variaveis:
#   quanto maior o CV, mais heterogenea e a variavel. malic_acid tem CV bem maior
#   que alcohol, ou seja, o acido malico varia muito mais entre os vinhos.
# - Os quartis mostram que 50% dos vinhos ficam entre Q1 e Q3 (intervalo IQR).

# ----------------------------------------------------------------------
# 1c) ANALISE PROBABILISTICA: distribuicao de probabilidade + calculos
# ----------------------------------------------------------------------
titulo("PARTE 1c - ANALISE PROBABILISTICA (Distribuicao Normal)")

# Escolha: modelar o teor alcoolico (alcohol) com uma Distribuicao Normal,
# pois o histograma e aproximadamente simetrico e em forma de sino.
x = df["alcohol"]
mu, sigma = x.mean(), x.std(ddof=1)
print(f"Parametros estimados: mu = {mu:.4f} | sigma = {sigma:.4f}")

# Teste de normalidade (Shapiro-Wilk): H0 = os dados seguem distribuicao normal
w, p_valor = stats.shapiro(x)
print(f"Shapiro-Wilk: W = {w:.4f} | p-valor = {p_valor:.4f}")
print("  -> " + ("Nao rejeita H0 (compativel com a Normal) ao nivel de 5%."
                 if p_valor > 0.05 else
                 "Rejeita H0 ao nivel de 5%, mas a Normal ainda e uma aproximacao razoavel."))

dist = stats.norm(loc=mu, scale=sigma)

# Calculo 1: P(X > 13)  -> vinhos com mais de 13 graus de alcool
p1 = 1 - dist.cdf(13)
# Calculo 2: P(12 < X < 14)  -> vinhos com teor entre 12 e 14
p2 = dist.cdf(14) - dist.cdf(12)
# Calculo 3: valor x tal que 90% dos vinhos tem teor ate x (percentil 90)
x90 = dist.ppf(0.90)
# Calculo 4: P(X < 12)
p4 = dist.cdf(12)

# Comparacao com a frequencia observada (empirica)
emp1 = (x > 13).mean()
emp2 = ((x > 12) & (x < 14)).mean()
emp4 = (x < 12).mean()

print(f"\nP(X > 13)       teorica = {p1:.4f} | observada = {emp1:.4f}")
print(f"P(12 < X < 14)  teorica = {p2:.4f} | observada = {emp2:.4f}")
print(f"P(X < 12)       teorica = {p4:.4f} | observada = {emp4:.4f}")
print(f"Percentil 90 (teorico): {x90:.2f} graus de alcool")

# Calculo 5: Probabilidade condicional (empirica)
# P(alcohol > 13 | malic_acid > 2)
cond = df[df["malic_acid"] > 2]
p_cond = (cond["alcohol"] > 13).mean()
print(f"\nP(alcohol > 13 | malic_acid > 2) = {p_cond:.4f} (empirica, n = {len(cond)})")

# Grafico: histograma (densidade) com curva normal e area de P(X > 13)
fig, ax = plt.subplots(figsize=(9, 5))
ax.hist(x, bins=12, density=True, alpha=0.6, edgecolor="black", label="Dados")
xs = np.linspace(x.min() - 1, x.max() + 1, 400)
ax.plot(xs, dist.pdf(xs), linewidth=2, label=f"Normal(mu={mu:.2f}, sigma={sigma:.2f})")
xs_area = np.linspace(13, x.max() + 1, 200)
ax.fill_between(xs_area, dist.pdf(xs_area), alpha=0.4, label=f"P(X>13) = {p1:.2%}")
ax.set_title("Teor alcoolico - ajuste da Distribuicao Normal")
ax.set_xlabel("alcohol")
ax.set_ylabel("Densidade")
ax.legend()
salvar("parte1c_normal_alcohol.png")

# Interpretacao (comentarios):
# - As probabilidades teoricas ficam proximas das frequencias observadas, o que
#   indica que a Normal e uma boa aproximacao para o teor alcoolico (o Shapiro-Wilk
#   rejeita a normalidade exata a 5%, pois a distribuicao e um pouco mais achatada).
# - Ex.: cerca de 50% dos vinhos teriam mais de 13 graus de alcool.
# - A probabilidade condicional mostra se ter mais acido malico esta associado
#   a um teor alcoolico maior ou menor.





# ======================================================================
# PARTE 2 - MACHINE LEARNING & MODELLING (K-means)
# ======================================================================




# ----------------------------------------------------------------------
# 2a) INTRODUCAO
# ----------------------------------------------------------------------
titulo("PARTE 2a - INTRODUCAO")
print(
    "Problema: temos 178 vinhos descritos por 13 caracteristicas quimicas e nao\n"
    "sabemos previamente a qual tipo cada um pertence (aprendizado NAO supervisionado).\n"
    "Solucao proposta: usar o algoritmo K-means para agrupar os vinhos mais\n"
    "semelhantes entre si, definir o numero ideal de grupos (Elbow + Silhouette)\n"
    "e interpretar o perfil quimico de cada grupo (teor alcoolico, acidez etc.)."
)

# ----------------------------------------------------------------------
# 2b) PRE-PROCESSAMENTO
# ----------------------------------------------------------------------
titulo("PARTE 2b - PRE-PROCESSAMENTO")

# 1) Nulos: verificar e tratar (se houver, remove as linhas)
print("Nulos antes:", int(df.isnull().sum().sum()))
df_ml = df.dropna().copy()
print("Nulos depois:", int(df_ml.isnull().sum().sum()), "| linhas:", len(df_ml))

# 2) Selecao de features: o dataset ja nao possui coluna de classe (rotulo),
#    entao usamos todas as variaveis quimicas. Para simplificar e facilitar a
#    interpretacao, poderiamos usar so um subconjunto; aqui mantemos as 13.
features = list(df_ml.columns)
print("Features utilizadas:", features)

# 3) Padronizacao (z-score): K-means usa distancia euclidiana, e variaveis
#    como 'proline' (centenas/milhares) dominariam variaveis como 'hue' (~1).
scaler = StandardScaler()
X = scaler.fit_transform(df_ml[features])
print("Media apos padronizar (~0):", np.round(X.mean(axis=0), 3))
print("Desvio apos padronizar (~1):", np.round(X.std(axis=0), 3))

# ----------------------------------------------------------------------
# 2c) AJUSTE DO K-MEANS (exemplo inicial com k=3 antes de escolher o melhor k)
# ----------------------------------------------------------------------
titulo("PARTE 2c - AJUSTE K-MEANS")
kmeans_teste = KMeans(n_clusters=3, init="k-means++", n_init=10,
                      random_state=RANDOM_STATE)
kmeans_teste.fit(X)
print("Ajuste inicial com k=3 -> inercia:", round(kmeans_teste.inertia_, 2))
print("Tamanho dos grupos:", np.bincount(kmeans_teste.labels_))

# ----------------------------------------------------------------------
# 2d) ELBOW E SILHOUETTE - numero ideal de grupos
# ----------------------------------------------------------------------
titulo("PARTE 2d - ELBOW E SILHOUETTE SCORE")
ks = range(2, 11)
inercias, silhuetas = [], []
for k in ks:
    km = KMeans(n_clusters=k, init="k-means++", n_init=10,
                random_state=RANDOM_STATE)
    rotulos = km.fit_predict(X)
    inercias.append(km.inertia_)
    silhuetas.append(silhouette_score(X, rotulos))

resultados = pd.DataFrame({"k": list(ks), "inercia": inercias,
                           "silhouette": silhuetas}).round(4)
print(resultados.to_string(index=False))

fig, eixos = plt.subplots(1, 2, figsize=(12, 4.5))
eixos[0].plot(list(ks), inercias, marker="o")
eixos[0].set_title("Metodo Elbow (cotovelo)")
eixos[0].set_xlabel("Numero de grupos (k)")
eixos[0].set_ylabel("Inercia (WCSS)")
eixos[1].plot(list(ks), silhuetas, marker="o", color="tab:green")
eixos[1].set_title("Silhouette Score")
eixos[1].set_xlabel("Numero de grupos (k)")
eixos[1].set_ylabel("Silhouette medio")
salvar("parte2d_elbow_silhouette.png")

k_silhouette = list(ks)[int(np.argmax(silhuetas))]
print(f"\nMelhor k pelo Silhouette: {k_silhouette} "
      f"(score = {max(silhuetas):.4f})")

# Escolha do k final: o cotovelo da curva de inercia e o maior silhouette
# apontam para k = 3 (o dataset e composto por 3 tipos/cultivares de vinho).
K_FINAL = 3
print(f"K escolhido: {K_FINAL}")

# ----------------------------------------------------------------------
# 2e) ANALISE DOS GRUPOS
# ----------------------------------------------------------------------
titulo("PARTE 2e - ANALISE DOS GRUPOS")
kmeans = KMeans(n_clusters=K_FINAL, init="k-means++", n_init=10,
                random_state=RANDOM_STATE)
df_ml["grupo"] = kmeans.fit_predict(X)

# Como os numeros dos grupos sao arbitrarios, reordenamos pelo teor alcoolico
# medio (grupo 1 = menor teor, grupo 3 = maior) para facilitar a leitura.
ordem = df_ml.groupby("grupo")["alcohol"].mean().sort_values().index
mapa = {antigo: novo + 1 for novo, antigo in enumerate(ordem)}
df_ml["grupo"] = df_ml["grupo"].map(mapa)

print("Quantidade de vinhos por grupo:")
print(df_ml["grupo"].value_counts().sort_index())

perfil = df_ml.groupby("grupo")[features].mean().round(2)
print("\nMedia de cada caracteristica por grupo (valores originais):")
print(perfil.T)

# Perguntas do enunciado
g_alcool = perfil["alcohol"].idxmax()
g_alcool_min = perfil["alcohol"].idxmin()
g_acido = perfil["malic_acid"].idxmax()
g_cinza_min = perfil["ash"].idxmin()
print("\n--- Respostas ---")
print(f"Qual teor alcoolico dos vinhos de cada grupo?")
for g in perfil.index:
    sub = df_ml[df_ml["grupo"] == g]["alcohol"]
    print(f"  Grupo {g}: media = {sub.mean():.2f} "
          f"(min {sub.min():.2f} | max {sub.max():.2f})")
print(f"Grupo com vinhos mais acidos (maior acido malico): Grupo {g_acido} "
      f"({perfil.loc[g_acido, 'malic_acid']:.2f})")
print(f"Grupo com maior teor alcoolico: Grupo {g_alcool}; "
      f"menor: Grupo {g_alcool_min}")
print(f"Grupo com menos cinzas (ash): Grupo {g_cinza_min} "
      f"({perfil.loc[g_cinza_min, 'ash']:.2f})")

# Descricao automatica de cada grupo (compara com a media geral)
print("\nPerfil resumido de cada grupo (destaques vs. media geral):")
z = (perfil - df_ml[features].mean()) / df_ml[features].std()
for g in perfil.index:
    altos = z.loc[g].sort_values(ascending=False).head(3).index.tolist()
    baixos = z.loc[g].sort_values().head(3).index.tolist()
    print(f"  Grupo {g}: mais ALTO em {altos} | mais BAIXO em {baixos}")

# Graficos: PCA 2D dos grupos e boxplot do teor alcoolico por grupo
pca = PCA(n_components=2, random_state=RANDOM_STATE)
X_pca = pca.fit_transform(X)
centros_pca = pca.transform(kmeans.cluster_centers_)
var_exp = pca.explained_variance_ratio_ * 100

fig, eixos = plt.subplots(1, 2, figsize=(13, 5))
for g in sorted(df_ml["grupo"].unique()):
    m = df_ml["grupo"].values == g
    eixos[0].scatter(X_pca[m, 0], X_pca[m, 1], label=f"Grupo {g}", alpha=0.75)
eixos[0].scatter(centros_pca[:, 0], centros_pca[:, 1], c="black", marker="X",
                 s=200, label="Centroides")
eixos[0].set_title("Grupos de vinhos (projecao PCA)")
eixos[0].set_xlabel(f"PC1 ({var_exp[0]:.1f}%)")
eixos[0].set_ylabel(f"PC2 ({var_exp[1]:.1f}%)")
eixos[0].legend()

dados_box = [df_ml[df_ml["grupo"] == g]["alcohol"] for g in sorted(df_ml["grupo"].unique())]
eixos[1].boxplot(dados_box)
eixos[1].set_xticklabels([f"Grupo {g}" for g in sorted(df_ml["grupo"].unique())])
eixos[1].set_title("Teor alcoolico por grupo")
eixos[1].set_ylabel("alcohol")
salvar("parte2e_grupos.png")

# Heatmap do perfil padronizado dos grupos
fig, ax = plt.subplots(figsize=(10, 4))
im = ax.imshow(z.values, cmap="coolwarm", aspect="auto", vmin=-2, vmax=2)
ax.set_xticks(range(len(features)))
ax.set_xticklabels(features, rotation=60, ha="right")
ax.set_yticks(range(len(z.index)))
ax.set_yticklabels([f"Grupo {g}" for g in z.index])
for i in range(z.shape[0]):
    for j in range(z.shape[1]):
        ax.text(j, i, f"{z.values[i, j]:.1f}", ha="center", va="center", fontsize=7)
fig.colorbar(im, label="Desvios-padrao em relacao a media geral")
ax.set_title("Perfil dos grupos (valores padronizados)")
salvar("parte2e_heatmap_perfil.png")

# Conclusao (comentarios):
# - O K-means separou os vinhos em 3 grupos com perfis quimicos distintos.
# - Grupo 1: menor teor alcoolico (~12,3), pouca cor e prolina baixa.
# - Grupo 2: teor alcoolico intermediario (~13,1), o mais acido (acido malico
#   ~3,3), cor muito intensa e poucos flavonoides.
# - Grupo 3: maior teor alcoolico (~13,7), muita prolina, flavonoides e fenois,
#   e menos acido malico.
# - O silhouette (~0,28) indica grupos moderadamente separados, com alguma
#   sobreposicao entre vinhos de perfis proximos.
# (Os numeros acima vem da execucao com random_state=42; confira nas tabelas.)

df_ml.to_csv("vinhos_com_grupos.csv", index=False)
print("\nArquivo 'vinhos_com_grupos.csv' salvo com a coluna 'grupo'.")
print("Fim da execucao.")
