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
    "sabemos previamente a qual tipo cada um pertence (aprendizado NAO supervisionado:\n"
    "nao existe uma coluna alvo y).\n"
    "Solucao proposta: usar o algoritmo K-means para agrupar os vinhos mais\n"
    "semelhantes entre si, definir o numero ideal de grupos (Elbow + Silhouette)\n"
    "e interpretar o perfil quimico de cada grupo (teor alcoolico, acidez etc.)."
)

# ----------------------------------------------------------------------
# 2b) PRE-PROCESSAMENTO
# ----------------------------------------------------------------------
titulo("PARTE 2b - PRE-PROCESSAMENTO")

# Tratamento de nulos
print("Nulos antes:", int(df.isnull().sum().sum()))
df_ml = df.dropna().copy()
print("Nulos depois:", int(df_ml.isnull().sum().sum()), "| linhas:", len(df_ml))

# Selecao de features
features = list(df_ml.columns)
print("Features utilizadas:", features)

# Padronizacao (StandardScaler)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df_ml[features])
print("Media apos padronizar (~0):", np.round(X_scaled.mean(axis=0), 3))
print("Desvio apos padronizar (~1):", np.round(X_scaled.std(axis=0), 3))

# Visualizacao inicial com PCA
pca = PCA(n_components=2, random_state=RANDOM_STATE)
X_pca = pca.fit_transform(X_scaled)
print("Variancia explicada pelo PCA (PC1, PC2):",
      np.round(pca.explained_variance_ratio_ * 100, 1), "%")

plt.figure(figsize=(7, 5))
plt.scatter(X_pca[:, 0], X_pca[:, 1], alpha=0.8)
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("Visualizacao inicial dos dados com PCA")
salvar("parte2b_pca_inicial.png")

# ----------------------------------------------------------------------
# 2c) AJUSTE DO K-MEANS
# ----------------------------------------------------------------------
titulo("PARTE 2c - AJUSTE K-MEANS")
# Definicao do modelo K-Means
modelo_base = KMeans(random_state=RANDOM_STATE, n_init=10)
print("Modelo base (k sera definido a seguir):", modelo_base)

# ----------------------------------------------------------------------
# 2d) ELBOW E SILHOUETTE SCORE - numero ideal de grupos
# ----------------------------------------------------------------------
titulo("PARTE 2d - ELBOW E SILHOUETTE SCORE")

# Tabela de inercia e silhouette para k = 2..10
ks = list(range(2, 11))
inercias, silhuetas = [], []
for k in ks:
    km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
    rotulos = km.fit_predict(X_scaled)
    inercias.append(km.inertia_)
    silhuetas.append(silhouette_score(X_scaled, rotulos))
print(pd.DataFrame({"k": ks, "inercia": inercias,
                    "silhouette": silhuetas}).round(4).to_string(index=False))


def k_cotovelo(ks, inercias):
    """Calcula o k do cotovelo."""
    quedas = -np.diff(inercias)          # quanto a inercia cai de k para k+1
    curvatura = quedas[:-1] - quedas[1:]  # desaceleracao da queda em cada k
    return ks[1:-1][int(np.argmax(curvatura))]


# --- Metodo do cotovelo (Elbow) ---
k_elbow = None
if USA_YELLOWBRICK:
    try:
        elbow = KElbowVisualizer(KMeans(random_state=RANDOM_STATE, n_init=10),
                                 k=(2, 10), timings=False)
        elbow.fit(X_scaled)
        elbow.finalize()
        k_elbow = elbow.elbow_value_
        salvar("parte2d_elbow.png")
    except Exception as erro:
        print("Yellowbrick indisponivel para o Elbow, usando matplotlib:", erro)
        plt.close("all")
        k_elbow = None
if k_elbow is None:
    k_elbow = k_cotovelo(ks, inercias)
    plt.figure(figsize=(8, 5))
    plt.plot(ks, inercias, marker="o")
    plt.axvline(k_elbow, linestyle="--", color="gray", label=f"cotovelo k={k_elbow}")
    plt.xlabel("Numero de grupos (k)")
    plt.ylabel("Inercia (distortion score)")
    plt.title("Metodo do cotovelo (Elbow) - K-Means")
    plt.legend()
    salvar("parte2d_elbow.png")
print(f"\nK sugerido pelo cotovelo: {k_elbow}")

# --- Silhouette: grafico de silhouette para k = 2..6 ---
fig, axes = plt.subplots(2, 3, figsize=(15, 8))
axes = axes.flatten()
for i, k in enumerate(range(2, 7)):
    km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
    feito = False
    if USA_YELLOWBRICK:
        try:
            vis = SilhouetteVisualizer(km, ax=axes[i])
            vis.fit(X_scaled)
            vis.finalize()
            feito = True
        except Exception:
            axes[i].clear()
    if not feito:
        rotulos = km.fit_predict(X_scaled)
        sil_amostras = silhouette_samples(X_scaled, rotulos)
        y_inf = 10
        for c in range(k):
            vals = np.sort(sil_amostras[rotulos == c])
            axes[i].fill_betweenx(np.arange(y_inf, y_inf + len(vals)), 0, vals, alpha=0.7)
            y_inf += len(vals) + 10
        axes[i].axvline(silhouette_score(X_scaled, rotulos), color="red", linestyle="--")
        axes[i].set_xlabel("Coeficiente de silhouette")
        axes[i].set_yticks([])
    axes[i].set_title(f"k = {k}")
fig.delaxes(axes[-1])  # remove o 6o espaco vazio
salvar("parte2d_silhouette_k2_a_k6.png")

# --- Escolha do k ---
k_silhouette = ks[int(np.argmax(silhuetas))]
print(f"K com maior silhouette medio: {k_silhouette} "
      f"(score = {max(silhuetas):.4f})")

# Escolha do k final (Elbow e Silhouette)
k_ideal = 3
print(f"K escolhido: {k_ideal}")

# ----------------------------------------------------------------------
# 2e) ANALISE DOS GRUPOS
# ----------------------------------------------------------------------
titulo("PARTE 2e - ANALISE DOS GRUPOS")
kmeans = KMeans(n_clusters=k_ideal, random_state=RANDOM_STATE, n_init=10)
kmeans_labels = kmeans.fit_predict(X_scaled)

# Reordenacao dos grupos por teor alcoolico medio
ordem = df_ml.assign(c=kmeans_labels).groupby("c")["alcohol"].mean().sort_values().index
mapa = {antigo: novo + 1 for novo, antigo in enumerate(ordem)}
kmeans_labels = np.array([mapa[l] for l in kmeans_labels])
centros_ordenados = kmeans.cluster_centers_[list(ordem)]  # mesma ordem dos grupos

df_kmeans = df_ml.copy()
df_kmeans["cluster_kmeans"] = kmeans_labels
print("Quantidade de vinhos por grupo:")
print(df_kmeans["cluster_kmeans"].value_counts().sort_index())

# Silhouette medio do modelo escolhido
sil_kmeans = silhouette_score(X_scaled, kmeans_labels)
print("\nSilhouette score do K-Means:", round(sil_kmeans, 4))

# --- Clusters no plano do PCA, com os centroides ---
centroides_pca = pca.transform(centros_ordenados)
plt.figure(figsize=(7, 5))
for g in sorted(df_kmeans["cluster_kmeans"].unique()):
    m = kmeans_labels == g
    plt.scatter(X_pca[m, 0], X_pca[m, 1], alpha=0.85, label=f"Grupo {g}")
plt.scatter(centroides_pca[:, 0], centroides_pca[:, 1], c="black", s=200,
            marker="X", label="Centroides")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title(f"K-Means com k={k_ideal} - visualizacao via PCA + centroides")
plt.legend()
salvar("parte2e_clusters_pca.png")

# --- Medias das features por cluster (valores originais) ---
medias_kmeans = df_kmeans.groupby("cluster_kmeans")[features].mean().round(2)
print("\nMedia de cada caracteristica por grupo (valores originais):")
print(medias_kmeans.T)

# --- Boxplots das features por cluster (dispersao) ---
n_cols = 3
n_rows = int(np.ceil(len(features) / n_cols))
fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4 * n_rows))
axes = axes.flatten()
for i, feature in enumerate(features):
    sns.boxplot(data=df_kmeans, x="cluster_kmeans", y=feature, ax=axes[i])
    axes[i].set_title(feature)
for j in range(i + 1, len(axes)):
    fig.delaxes(axes[j])
salvar("parte2e_boxplots_por_cluster.png")

# --- Centroides no espaco escalado ---
centroides = pd.DataFrame(
    centros_ordenados, columns=features,
    index=[f"grupo_{g}" for g in range(1, k_ideal + 1)]
)
print("\nCentroides (espaco padronizado):")
print(centroides.T.round(2))

# --- Respostas as perguntas do enunciado ---
g_acido = medias_kmeans["malic_acid"].idxmax()
g_alcool_max = medias_kmeans["alcohol"].idxmax()
g_alcool_min = medias_kmeans["alcohol"].idxmin()
g_cinza_min = medias_kmeans["ash"].idxmin()
print("\n--- Respostas ---")
print("Qual teor alcoolico dos vinhos de cada grupo?")
for g in medias_kmeans.index:
    sub = df_kmeans[df_kmeans["cluster_kmeans"] == g]["alcohol"]
    print(f"  Grupo {g}: media = {sub.mean():.2f} "
          f"(min {sub.min():.2f} | max {sub.max():.2f})")
print(f"Grupo com vinhos mais acidos (maior acido malico): Grupo {g_acido} "
      f"({medias_kmeans.loc[g_acido, 'malic_acid']:.2f})")
print(f"Grupo com maior teor alcoolico: Grupo {g_alcool_max}; "
      f"menor: Grupo {g_alcool_min}")
print(f"Grupo com menos cinzas (ash): Grupo {g_cinza_min} "
      f"({medias_kmeans.loc[g_cinza_min, 'ash']:.2f})")

# Destaques de cada grupo em relacao a media geral (em desvios-padrao)
z = (medias_kmeans - df_kmeans[features].mean()) / df_kmeans[features].std()
print("\nPerfil resumido de cada grupo (destaques vs. media geral):")
for g in z.index:
    altos = z.loc[g].sort_values(ascending=False).head(3).index.tolist()
    baixos = z.loc[g].sort_values().head(3).index.tolist()
    print(f"  Grupo {g}: mais ALTO em {altos} | mais BAIXO em {baixos}")

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

# Exportacao do resultado
df_kmeans.to_csv("vinhos_com_grupos.csv", index=False)
print("\nArquivo 'vinhos_com_grupos.csv' salvo com a coluna 'cluster_kmeans'.")
print("Fim da execucao.")


