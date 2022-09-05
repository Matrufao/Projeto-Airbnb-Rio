#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd    # trabalhar com dados
import pathlib      # percorrer toda uma pasta
import numpy as np    # tratamento de numero, series, variaveis
import seaborn as sns   # biblioteca de graficos
import matplotlib.pyplot as plt    # biblioteca de graficos
import plotly.express as px
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.model_selection import train_test_split

print("PRONTO")


# ### Importar Bibliotecas e Bases de Dados

# In[2]:


meses = {'jan': 1, 'fev':2, 'mar':3, 'abr': 4, 'mai':5, 'jun': 6, 'jul': 7, 'ago': 8, 'set': 9, 'out': 10, 'nov': 11, 'dez': 12}

caminho_bases = pathlib.Path(r'C:\Users\Samsung\Desktop\TREINAMENTO\python\projetoAirbnb\dataset')

base_airbnb = pd.DataFrame()

for arquivo in caminho_bases.iterdir():
    nome_mes = arquivo.name[:3]
    mes = meses[nome_mes]
    
    ano = arquivo.name[-8:]
    ano = int(ano.replace('.csv', ''))
    
    df = pd.read_csv(caminho_bases / arquivo.name, low_memory=False)
    df['ano'] = ano
    df['mes'] = mes
    base_airbnb = pd.concat([df, base_airbnb])


# In[3]:


# visualizar dados
base_d_dados = base_airbnb
display(base_d_dados)


# In[4]:


# espaço para contar valor

print(base_d_dados[["experiences_offered"]].value_counts())


# In[5]:


# espaço para comparar colunas

print((base_d_dados["host_listings_count"]==base_d_dados["host_total_listings_count"]).value_counts())


# ### Se tivermos muitas colunas, já vamos identificar quais colunas podemos excluir
# 
# 

# In[6]:


print(list(base_d_dados.columns))  # listar name das colunas
# a primeiro momento vai ser deletado as colunas que não vão ser utilizadas:
# tipo de colunas deletadas: 

base_d_dados.head(1000).to_csv("1003 primeiro registro.csv", sep=";")  # criarplanilha excel com os 1000 primeiro registro para ver qual
# dados vai ser util


# # Colunas irrelevantes:
# numero ou texto com informação aleatorio o qual afetaria a analise
# last_scraped: ja foi criano duas nova colunas com as mesama informação (o ano e o mês refente ao dados)
# colunas com texto livre pois não havera analise com textos
# colinas com textos iguais
# colunas com informações similares
# 
# colunas mantida: 'host_response_time','host_response_rate','host_is_superhost','host_listings_count','latitude','longitude','property_type','room_type','accommodates','bathrooms','bedrooms','beds','bed_type','amenities','price','security_deposit','cleaning_fee','guests_included','extra_people','minimum_nights','maximum_nights','number_of_reviews','number_of_reviews_ltm','review_scores_rating','review_scores_accuracy','review_scores_cleanliness','review_scores_checkin','review_scores_communication','review_scores_location','review_scores_value','instant_bookable','is_business_travel_ready','cancellation_policy','mes','ano'

# In[7]:


# filtrando as colunas que vamos usar

colunas = ['host_response_time','host_response_rate','host_is_superhost','host_listings_count','latitude','longitude','property_type','room_type','accommodates','bathrooms','bedrooms','beds','bed_type','amenities','price','security_deposit','cleaning_fee','guests_included','extra_people','minimum_nights','maximum_nights','number_of_reviews','number_of_reviews_ltm','review_scores_rating','review_scores_accuracy','review_scores_cleanliness','review_scores_checkin','review_scores_communication','review_scores_location','review_scores_value','instant_bookable','is_business_travel_ready','cancellation_policy','mes','ano'
]
base_d_dados = base_d_dados.loc[:,colunas]  #  ":" todas as linhas
display(base_d_dados)


# ### Tratar Valores Faltando

# In[8]:


# informações da base de dados

base_d_dados.info()

# analisar os valores faltante

print(base_d_dados.isnull().sum())


# In[9]:


# para colunas com dados superiores a 300000 serão removidos, subtituir esses valores por um numero "aleatorio" tornaria todo o 
#trabalho invalido, afetando a previsão

for coluna in base_d_dados:
    if base_d_dados[coluna].isnull().sum() > 300000:
        base_d_dados = base_d_dados.drop(coluna, axis=1)
print(base_d_dados.isnull().sum())


# In[10]:


# remover todas as tinha que tenha algum valor vazio

base_d_dados = base_d_dados.dropna()
print(base_d_dados.isnull().sum())


# ### Verificar Tipos de Dados em cada coluna

# In[11]:


print(base_d_dados.dtypes)
print("-"*60)
print(base_d_dados.iloc[0])   # mostrar todo a linha do indice 0


# In[12]:



# dados errados: price, extra_people

# tratanto coluna price : mudar vairavel, trocar $ e subistituir "," por "." 
base_d_dados["price"] = base_d_dados["price"].str.replace("$","")      # str: tratar string de texto
base_d_dados["price"] = base_d_dados["price"].str.replace(",","")      
base_d_dados["price"] = base_d_dados["price"].astype(np.float32) # transfomar texto em numero

# tratanto coluna extra_people : mudar vairavel, trocar $ e subistituir "," por "." 
base_d_dados["extra_people"] = base_d_dados["extra_people"].str.replace("$","")      # str: tratar string de texto
base_d_dados["extra_people"] = base_d_dados["extra_people"].str.replace(",","")      
base_d_dados["extra_people"] = base_d_dados["extra_people"].astype(np.float32) # transfomar texto em numero

# tratanto coluna mes : mudar vairavel  
df['ano'] = df['ano'].astype(int) # transfomar texto em numero inteiro
df['mes'] = df['mes'].astype(int) # transfomar texto em numero inteiro


# In[13]:


print(base_d_dados.dtypes)


# ### Análise Exploratória e Tratar Outliers

# In[14]:


# verificar correlação com um mapa de calor

# print(base_d_dados.corr())
plt.figure(figsize=(15, 10))
sns.heatmap(base_d_dados.corr(), annot=True, cmap='Reds')


# # Definição de função para analise de outliers
# 
# outliers são dados que se diferenciam drasticamente de todos os outros. Em outras palavras, um outlier é um valor que foge da normalidade e que pode (e provavelmente irá) causar anomalias nos resultados obtidos por meio de algoritmos e sistemas de análise

# In[15]:


# definir limite para colunas
def limites(coluna):
    q1 = coluna.quantile(0.25)
    q3 = coluna.quantile(0.75)
    amplitude = q3 - q1
    return q1 - 1.5 * amplitude, q3 + 1.5 *amplitude


# In[16]:


# definir função para diagrama de caixa
def diagrama_cx(coluna):
    fig, (ax1, ax2) = plt.subplots(1, 2)     # configuração do grafico
    fig.set_size_inches(15, 5)              # configuração do grafico
    sns.boxplot(x=coluna, ax=ax1)  
    ax2.set_xlim(limites(coluna))
    sns.boxplot(x=coluna, ax=ax2)
    
# definir função para histrograma
def histrograma(coluna):
    plt.figure(figsize=(15,5))                     # configuração do grafico
    sns.histplot(coluna, bins=50,  kde=True)     # grafico hitrograma 

# definir função para excluir outliers
def excluir_outliers(base_d_dados, nome_coluna):
    qtd_linha = base_d_dados.shape[0]
    lim_inf, lim_sup = limites(base_d_dados[nome_coluna])
    base_d_dados = base_d_dados.loc[(base_d_dados[nome_coluna] >= lim_inf) & (base_d_dados[nome_coluna] <= lim_sup), :]
    linhas_removidas = qtd_linha - base_d_dados.shape[0]
    return base_d_dados, linhas_removidas

def grafico_barra(coluna):  
    plt.figure(figsize=(15, 5))
    ax = sns.barplot(x=coluna.value_counts().index, y=coluna.value_counts())
    ax.set_xlim(limites(coluna))


# ### Analise de outliers
# para analise o objetivo do projeto sera para imóveis comum então sera removidos outliers onde podera atraplhar para analise, como quartos e banheiros em grande quantidade ou alto preço onde seria necessario imóveis de alto padrão

# # Price

# In[17]:


# visualização dos graficos
diagrama_cx(base_d_dados["price"])
histrograma(base_d_dados["price"])


# 

# In[18]:


base_d_dados, linhas_removida = excluir_outliers(base_d_dados, "price")
print('{} linhas removidas'.format(linhas_removida))


# In[19]:


# visualização dos graficos
diagrama_cx(base_d_dados["price"])
histrograma(base_d_dados["price"])


# ### extra_people

# In[20]:


diagrama_cx(base_d_dados['extra_people'])
histrograma(base_d_dados['extra_people'])


# In[21]:


# Excluindo outliers do extra_people
base_d_dados, linhas_removida = excluir_outliers(base_d_dados, "extra_people")
print('{} linhas removidas'.format(linhas_removida))


# In[22]:


diagrama_cx(base_d_dados["price"])
histrograma(base_d_dados["price"])


# ### host_listings_count

# In[23]:


diagrama_cx(base_d_dados["host_listings_count"])
grafico_barra(base_d_dados["host_listings_count"])


# In[24]:


# Excluindo outliers do host_listings_count
base_d_dados, linhas_removida = excluir_outliers(base_d_dados, "host_listings_count")
print('{} linhas removidas'.format(linhas_removida))


# In[25]:


# exbir grafico
diagrama_cx(base_d_dados["host_listings_count"])
grafico_barra(base_d_dados["host_listings_count"])


# ### accommodates

# In[26]:


diagrama_cx(base_d_dados["accommodates"])
grafico_barra(base_d_dados["accommodates"])


# vamos excluir os outliers dessa coluna porque apartamentos que acomodam mais de 9 pessoas nosso objetivo é para imóveis comun

# In[27]:


# Excluindo outliers do accommodates
base_d_dados, linhas_removida = excluir_outliers(base_d_dados, "accommodates")
print('{} linhas removidas'.format(linhas_removida))


# In[28]:


# visualizar grafico 
diagrama_cx(base_d_dados["accommodates"])
grafico_barra(base_d_dados["accommodates"])


# ### bathrooms

# In[29]:


diagrama_cx(base_d_dados["bathrooms"])
grafico_barra(base_d_dados.bathrooms)


# In[30]:


# Excluindo outliers do bathrooms
base_d_dados, linhas_removida = excluir_outliers(base_d_dados, "bathrooms")
print('{} linhas removidas'.format(linhas_removida))


# In[31]:


diagrama_cx(base_d_dados.bathrooms)
grafico_barra(base_d_dados.bathrooms)


# ### bedrooms

# In[32]:


diagrama_cx(base_d_dados.bedrooms)
grafico_barra(base_d_dados.bedrooms)


# In[33]:


# Excluindo outliers do bedrooms
base_d_dados, linhas_removida = excluir_outliers(base_d_dados, "bedrooms")
print('{} linhas removidas'.format(linhas_removida))


# In[34]:


# visualizar graficos
diagrama_cx(base_d_dados["bedrooms"])
grafico_barra(base_d_dados.bedrooms)


# ### beds

# In[35]:


diagrama_cx(base_d_dados["beds"])
grafico_barra(base_d_dados.beds)


# In[36]:


# Excluindo outliers do beds
base_d_dados, linhas_removida = excluir_outliers(base_d_dados, "beds")
print('{} linhas removidas'.format(linhas_removida))


# In[37]:


# visualizar graficos
diagrama_cx(base_d_dados["beds"])
grafico_barra(base_d_dados.beds)


# ### guests_included

# In[38]:


diagrama_cx(base_d_dados.guests_included)
histrograma(base_d_dados.guests_included)
print(limites(base_d_dados.guests_included))


# Essa feature da análise será removida pois ha hipotese que os usuários do airbnb usam muito o valor padrão do airbnb como 1 guest included. Isso pode levar o nosso modelo a considerar uma feature que na verdade não é essencial para a definição do preço, por isso para não afetar no final do projeto essa featura sera removida

# In[39]:


base_d_dados = base_d_dados.drop('guests_included', axis=1)


# ### minimum_nights

# In[40]:


diagrama_cx(base_d_dados.minimum_nights)
histrograma(base_d_dados.minimum_nights)


# In[41]:


# Excluindo outliers do minimum_nights
base_d_dados, linhas_removida = excluir_outliers(base_d_dados, "minimum_nights")
print('{} linhas removidas'.format(linhas_removida))


# In[42]:


# visualizar graficos
diagrama_cx(base_d_dados.minimum_nights)
grafico_barra(base_d_dados.minimum_nights)


# ### maximum_nights

# In[43]:


diagrama_cx(base_d_dados.maximum_nights)
histrograma(base_d_dados.maximum_nights)


# Essa coluna de maximum_nights também irar afetar nossa analise levando em consideração que não vamos ter um limite maximo de noite e que normamente este campo não é preenchido

# In[44]:


# excluindo coluna
base_d_dados = base_d_dados.drop('maximum_nights', axis=1)
base_d_dados.shape


# ### number_of_reviews

# In[45]:


diagrama_cx(base_d_dados.number_of_reviews)
histrograma(base_d_dados.number_of_reviews)


# Se excluirmos os outliers, vamos excluir as pessoas que tem a maior quantidade de reviews (o que normalmente são os hosts que têm mais aluguel). Isso pode impactar muito negativamente o nosso modelo Pensando no nosso objetivo, se eu tenho um imóvel parado e quero colocar meu imóvel lá, é claro que eu não tenho review nenhuma. Então talvez tirar essa característica da análise pode na verdade acabar ajudando.

# In[46]:


# excluindo coluna
base_d_dados = base_d_dados.drop('number_of_reviews', axis=1)
base_d_dados.shape


# ### Tratamento de Colunas de Valores de Texto

# In[47]:


print(base_d_dados.property_type.value_counts())

plt.figure(figsize=(15, 5))
grafico = sns.countplot('property_type', data=base_d_dados)
grafico.tick_params(axis='x', rotation=90)


# todos os tipos de propriedade que têm menos de 2.000 propriedades na base de dados, eu vou agrupar em um grupo chamado "outros"

# In[48]:


# Agrupando todos os tipo de imoveis que tem menos de 2000
tabela_casa = base_d_dados.property_type.value_counts()
colunas_agrupar = []

for tipo in tabela_casa.index:
    if tabela_casa[tipo] < 2000:
        colunas_agrupar.append(tipo)
print(colunas_agrupar)
for tipo in colunas_agrupar:
    base_d_dados.loc[base_d_dados["property_type"]==tipo, "property_type"] = "outros"


# In[49]:


plt.figure(figsize=(15, 5))
grafico = sns.countplot('property_type', data=base_d_dados)
grafico.tick_params(axis='x', rotation=90)


# ### room_type

# In[50]:


print(base_d_dados.room_type.value_counts())

plt.figure(figsize=(15, 5))
grafico = sns.countplot('room_type', data=base_d_dados)
grafico.tick_params(axis='x', rotation=90)


# ### bed_type

# In[51]:



print(base_d_dados['bed_type'].value_counts())

# agrupando categorias de cancellation_pollicy
tabela_bed = base_d_dados['bed_type'].value_counts()
colunas_agrupar = []

for tipo in tabela_bed.index:
    if tabela_bed[tipo] < 10000:
        colunas_agrupar.append(tipo)
print("-"*60)
print(colunas_agrupar)

for tipo in colunas_agrupar:
    base_d_dados.loc[base_d_dados['bed_type']==tipo, 'bed_type'] = 'outros'

print(base_airbnb['bed_type'].value_counts())
plt.figure(figsize=(15, 5))
grafico = sns.countplot('bed_type', data=base_d_dados)
grafico.tick_params(axis='x', rotation=90)


# ### cancellation_policy

# In[52]:


print(base_d_dados['cancellation_policy'].value_counts())

# agrupando categorias de cancellation_pollicy
tabela_cancellation = base_d_dados['cancellation_policy'].value_counts()
colunas_agrupar = []

for tipo in tabela_cancellation.index:
    if tabela_cancellation[tipo] < 10000:
        colunas_agrupar.append(tipo)
print("-"*60)
print(colunas_agrupar)

for tipo in colunas_agrupar:
    base_d_dados.loc[base_d_dados['cancellation_policy']==tipo, 'cancellation_policy'] = 'strict'

print(base_airbnb['cancellation_policy'].value_counts())
plt.figure(figsize=(15, 5))
grafico = sns.countplot('cancellation_policy', data=base_d_dados)
grafico.tick_params(axis='x', rotation=90)


# ### - amenities 
# 
# Como temos uma diversidade muito grande de amenities e, às vezes, as mesmas amenities podem ser escritas de forma diferente, vamos avaliar a quantidade de amenities como o parâmetro para o nosso modelo.

# In[53]:


print(base_d_dados['amenities'].iloc[1].split(','))
print(len(base_d_dados['amenities'].iloc[1].split(',')))

base_d_dados['n_amenities'] = base_d_dados['amenities'].str.split(',').apply(len)


# In[54]:


base_d_dados = base_d_dados.drop('amenities', axis=1)
base_d_dados.shape


# In[55]:


# visualizar grafico
diagrama_cx(base_d_dados['n_amenities'])
grafico_barra(base_d_dados['n_amenities'])


# In[56]:


base_d_dados.n_amenities.value_counts()
# agora a coluna de amenities virou coluna com valor numerico, agora da pra excluir os outliers


# In[57]:


# excluir outliers
base_d_dados, linhas_removidas = excluir_outliers(base_d_dados, 'n_amenities')
print('{} linhas removidas'.format(linhas_removidas))


# In[58]:


# visualizar grafico apos excluir outliers
diagrama_cx(base_d_dados['n_amenities'])
grafico_barra(base_d_dados['n_amenities'])


#  ### Visualização de Mapa das Propriedades

# Vamos criar um mapa que exibe um pedaço da nossa base de dados aleatório (50.000 propriedades) para ver como as propriedades estão distribuídas pela cidade e também identificar os locais de maior preço

# In[67]:


amostra = base_d_dados.sample(n=50000)
centro_mapa = {'lat':amostra.latitude.mean(), 'lon':amostra.longitude.mean()}
mapa = px.density_mapbox(amostra, lat='latitude', lon='longitude',z='price', radius=2.5,
                        center=centro_mapa, zoom=10,
                        mapbox_style='stamen-terrain')
mapa.show()


# ### Encoding
# Ajustar as features para facilitar o trabalho do modelo futuro (features de categoria, true e false, etc.)

# In[98]:


# Features de Valores True ou False, vamos substituir True por 1 e False por 0.

colunas_tf = ['host_is_superhost', 'instant_bookable', 'is_business_travel_ready']
base_d_dados_cod = base_d_dados.copy()
for coluna in colunas_tf:
    base_d_dados_cod.loc[base_d_dados_cod[coluna]=='t', coluna] = 1
    base_d_dados_cod.loc[base_d_dados_cod[coluna]=='f', coluna] = 0


# Features de Categoria (features em que os valores da coluna são textos) vamos utilizar o método de encoding de variáveis dummies

# In[99]:


colunas_categorias = ['property_type', 'room_type', 'bed_type', 'cancellation_policy']
base_d_dados_cod = pd.get_dummies(data=base_d_dados_cod, columns=colunas_categorias)
display(base_d_dados_cod.head())


# ### Modelo de Previsão

# Métricas de Avaliação
# Vamos usar aqui o R² que vai nos dizer o quão bem o nosso modelo consegue explicar o preço. Isso seria um ótimo parâmetro para ver o quão bom é nosso modelo
# -> Quanto mais próximo de 100%, melhor
# 
# RSME: Vou calcular também o Erro Quadrático Médio, que vai mostrar para gente o quanto o nosso modelo está errando.
# -> Quanto menor for o erro, melhor

# In[100]:


# definfir função para avalisar o modelo / quanto maus proximo de 100% melhor
def avaliar_modelo(nome_modelo, y_teste, previsao):
    r2 = r2_score(y_teste, previsao)
    RSME = np.sqrt(mean_squared_error(y_teste, previsao))
    return f'Modelo {nome_modelo}:\nR²:{r2:.2%}\nRSME:{RSME:.2f}'


# Escolha dos Modelos a Serem Testados
# 
# RandomForest
# 
# LinearRegression
# 
# Extra Tree
# 
# Estamos querendo calcular o preço, portanto, queremos prever um valor numérico.
# 
# Mas na dúvida, esses 3 modelos que usamos aqui são bem bons para muitos problemas de Regressão.

# In[63]:


modelo_rf = RandomForestRegressor()
modelo_lr = LinearRegression()
modelo_et = ExtraTreesRegressor()

modelos = {'RandomForest': modelo_rf,
          'LinearRegression': modelo_lr,
          'ExtraTrees': modelo_et,
          }

y = base_d_dados_cod['price']
X = base_d_dados_cod.drop('price', axis=1)


# In[64]:


# Separa os dados em treino e teste + Treino do Modelo
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=10)

for nome_modelo, modelo in modelos.items():
    #treinar
    modelo.fit(X_train, y_train)
    #testar
    previsao = modelo.predict(X_test)
    print(avaliar_modelo(nome_modelo, y_test, previsao))


# ### Análise do Melhor Modelo

# # O melhor modelo definido foi o modelo ExtraTrees
# tendo o maior valor nas ambas porcetagem metricas (R², RSME) comparado ao modelo Ramdom Forest e Linear Regression.
# resultado Modelo ExtraTrees:
# R²:97.38%
# RSME:42.95

# ### Ajustes e Melhorias no Melhor Modelo

# analise das colunas mais importantes usando no modelo, como por exempo a coluna is_business_travel_ready não é importante, tanto que não é utilizado no modelo, as colunas bedrooms,latitude e longitude são as mais importes pelo fato de estar ligado com a localização e quantas pessoas hospeda.
# 
# 

# In[85]:



importancia_freatures = pd.DataFrame(modelo_et.feature_importances_, X_train.columns)
importancia_freatures = importancia_freatures.sort_values(by=0, ascending=False)
display(importancia_freatures)


# In[102]:


plt.figure(figsize=(15, 5))
ax = sns.barplot(x=importancia_freatures.index, y=importancia_freatures[0])
ax.tick_params(axis="x", rotation=90)


# ultimo resultado: Modelo ExtraTrees:
# R²:97.38%
# RSME:42.95, agora remover colunas irrelevantes para deixar o modelo mais rapido e tentar deixar mais preciso

# In[101]:


base_d_dados_cod = base_d_dados_cod.drop("is_business_travel_ready", axis=1)

y = base_d_dados_cod['price']
X = base_d_dados_cod.drop('price', axis=1)

# Separa os dados em treino e teste + Treino do Modelo
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=10)

modelo_et.fit(X_train, y_train)
previsao = modelo.predict(X_test)
print(avaliar_modelo(nome_modelo, y_test, previsao))


# resultado apos remover a coluna is_business_travel_ready: Modelo ExtraTrees:
# R²:97.38%
# RSME:42.95, não mudou porem como cotém uma coluna a menos deixa o modelo mais rapido

# ## Deploy do projeto
# Salvar o modelo treinandoDisponibilizar o projeto para usuarios, criando arquivo joblib para ser usado para fazer previsões

# In[104]:


# Salvar base de dados tratada
X["price"] = y
X.to_csv("dados.csv")


# In[106]:


# exportar modelo de previsão
import joblib
joblib.dump(modelo_et, "modelo.joblib")

