import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import preprocessing

# Aplicação web com Streamlit
st.title("Previsão de Intenção de Compras Online")

# Carregar o dataset
shoppers_intention = pd.read_csv('./online_shoppers_intention.csv', sep=';', decimal=',')

# Limpeza dos dados
shoppers_intention.drop_duplicates(inplace=True)

label_encoder = preprocessing.LabelEncoder()
shoppers_intention['Weekend']= label_encoder.fit_transform(shoppers_intention['Weekend'])

shoppers_intention= pd.get_dummies(shoppers_intention, columns = ['Month','VisitorType'])

shoppers_intention['Administrative_Duration'] = shoppers_intention['Administrative_Duration'].str.replace('.', '').astype(float)
shoppers_intention['BounceRates'] = shoppers_intention['BounceRates'].astype(float)
shoppers_intention['ExitRates'] = shoppers_intention['ExitRates'].astype(float)
shoppers_intention['SpecialDay'] = shoppers_intention['SpecialDay'].astype(float)
shoppers_intention['PageValues'] = shoppers_intention['PageValues'].str.replace('.', '').astype(float)
shoppers_intention['ProductRelated_Duration'] = shoppers_intention['ProductRelated_Duration'].str.replace('.', '').astype(float)
shoppers_intention['Informational_Duration'] = shoppers_intention['Informational_Duration'].str.replace('.', '').astype(float)

# Treinamento do modelo
X = shoppers_intention.drop(columns=['Revenue'], axis=0)
y = shoppers_intention['Revenue'].values.ravel()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

#dados dos usuários com a função
def get_user_date():

  Administrative = st.sidebar.slider("Número de páginas visitadas pelo usuário para atividades relacionadas ao gerenciamento de contas de usuário",0, 27, 1)

  Administrative_Duration = st.sidebar.slider("Tempo gasto nas páginas administrativas pelo usuário",  0, 10000000000, 1983333333)

  Informational = st.sidebar.slider("Número de páginas visitadas pelo usuário sobre o site", 0, 24, 1)

  Informational_Duration = st.sidebar.slider("Tempo gasto em páginas informativas pelo usuário", 0, 10000000000, 1000000)

  ProductRelated = st.sidebar.slider("Número de páginas relacionadas ao produto visitadas pelo usuário", 0, 705, 20)

  ProductRelated_Duration = st.sidebar.slider("Tempo gasto pelo usuário nas páginas relacionadas ao produto", 0, 10000000000, 5450833333)

  BounceRates = st.sidebar.slider("Taxa média de rejeição das páginas visitadas pelo usuário", 0, 100000000, 1000000)

  ExitRates = st.sidebar.slider("Taxa média de saída das páginas visitadas pelo usuário", 0, 200000000, 1000000)

  PageValues = st.sidebar.slider("Valor médio das páginas visitadas pelo usuário", 0, 10000000000, 9836358025)

  SpecialDay = st.sidebar.slider("Proximidade do dia da visita com um evento especial", 0.0, 1.0, 0.0)

  OperatingSystems = st.sidebar.slider("Sistema Operacional do visitante", 1, 8, 3)

  Browser = st.sidebar.slider("Navegador do visitante", 1, 13, 2)

  Region = st.sidebar.slider("Região geográfica a partir da qual a sessão foi iniciada pelo visitante", 1, 9, 5)

  TrafficType = st.sidebar.slider("Fonte de tráfego através da qual o usuário entrou no site", 1, 20, 2)

  Weekend = st.sidebar.slider("Define se o usuário visitou o site no final de semana", 0, 1, 0)

  Month_Aug  = st.sidebar.slider("Define se o usuário visitou o site no mês de Agosto", 0, 1, 0)  

  Month_Dec  = st.sidebar.slider("Define se o usuário visitou o site no mês de Dezembro", 0, 1, 0)  

  Month_Feb  = st.sidebar.slider("Define se o usuário visitou o site no mês de Fevereiro", 0, 1, 0)  

  Month_Jul  = st.sidebar.slider("Define se o usuário visitou o site no mês de Julho", 0, 1, 0)  

  Month_June = st.sidebar.slider("Define se o usuário visitou o site no mês de Junhoo", 0, 1, 0)  

  Month_Mar  = st.sidebar.slider("Define se o usuário visitou o site no mês de Março", 0, 1, 0)  

  Month_May  = st.sidebar.slider("Define se o usuário visitou o site no mês de Maio", 0, 1, 0)  

  Month_Nov  = st.sidebar.slider("Define se o usuário visitou o site no mês de Novembro", 0, 1, 0)  

  Month_Oct  = st.sidebar.slider("Define se o usuário visitou o site no mês de Outubro", 0, 1, 0)  

  Month_Sep  = st.sidebar.slider("Define se o usuário visitou o site no mês de Setembro", 0, 1, 0)  

  VisitorType_New_Visitor  = st.sidebar.slider("Define se o usuário é novo visitante", 0, 1, 0) 

  VisitorType_Other        = st.sidebar.slider("Define se o usuário é outro tipo de visitante", 0, 1, 0)  
  
  VisitorType_Returning_Visitor  = st.sidebar.slider("Define se o usuário é um visitante que já conhece o site", 0, 1, 0)

    #dicionário para receber informações

  user_data = {'Administrative': Administrative,

    'Administrative_Duration': Administrative_Duration,

    'Informational': Informational,

    'Informational_Duration': Informational_Duration,

    'ProductRelated': ProductRelated,

    'ProductRelated_Duration': ProductRelated_Duration,

    'BounceRates': BounceRates,

    'ExitRates': ExitRates,

    'PageValues': PageValues,

    'SpecialDay': SpecialDay,

    'OperatingSystems': OperatingSystems,

    'Browser': Browser,

    'Region': Region,

    'TrafficType': TrafficType,

    'Weekend': Weekend,

    'Month_Aug': Month_Aug,

    'Month_Dec': Month_Dec,

    'Month_Feb': Month_Feb,

    'Month_Jul': Month_Jul,

    'Month_June': Month_June,

    'Month_Mar': Month_Mar,

    'Month_May': Month_May,

    'Month_Nov': Month_Nov,

    'Month_Oct': Month_Oct,

    'Month_Sep': Month_Sep,

    'VisitorType_New_Visitor': VisitorType_New_Visitor,

    'VisitorType_Other': VisitorType_Other,

    'VisitorType_Returning_Visitor': VisitorType_Returning_Visitor

    }

  features = pd.DataFrame(user_data, index=[0])

  return features

user_input_variables = get_user_date()

clf_random = RandomForestClassifier(n_estimators=100, max_depth=10, min_samples_split=2, min_samples_leaf=1,  random_state=42)
clf_random.fit(X_train, y_train)

# Prever e avaliar o modelo
y_pred = clf_random.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)*100

#acurácia do modelo
st.subheader('Acurácia do modelo em %:')
st.write(accuracy)

# Prever com base nos dados do usuário
prediction = clf_random.predict(user_input_variables)

# Mostrar o resultado
st.subheader('Previsão de Intenção de Compra:')
if prediction[0] == 1:
    st.write("Cliente REALIZARÁ a compra!")
else:
    st.write("Cliente NÃO realizará a compra.")
