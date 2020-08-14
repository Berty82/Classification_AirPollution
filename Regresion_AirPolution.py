
'''
2020 - MAGV 
Python 3.6.8 (default, Oct  9 2019, 14:04:01) 
[GCC 5.4.0 20160609] on linux

Este conjunto de datos contiene 9358 muestras, recogidas en orden temporal, de variables
(características) asociadas a la calidad del aire. 

Uno de los parámetros más importantes a la hora de evaluar la calidad del aire es el
dióxido de nitróngeno (NO2). Fuente: Agencia Europea de Medio Ambiente, directiva europea 
(2008/50/EC) contrastada con la guía de la organización mundial de la salud.
https://www.eea.europa.eu/themes/air/air-quality-concentrations/air-quality-standards

Apoyándome en esta información escojo el NO2 para el problema de regresión.

'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import seaborn as sns

import pdb


# ===========					Functions 			==============================
def remove_outlier(Data, col):
    Data[col] = Data.groupby('Date')[col].transform(lambda x: x.fillna(x.mean()))

#---------------------------------------------------------------------------------


#################			Carga de datos y preprocesamiento de datos 		######

Data = pd.read_csv('AirQualityUCI.csv',sep=',', delimiter=";",decimal=",")

#Data.head()
Data = Data.drop(["Unnamed: 15","Unnamed: 16"], axis=1)
#Data.info()
Data.isnull().any()
Data.isnull().sum()

# Elimino la fecha para trabajar más cómodo
Data.dropna(inplace=True)
Data.set_index("Date", inplace=True)
Data.index = pd.to_datetime(Data.index)
type(Data.index)

# Para evitar problemas paso a un valor entero los valores de la variable temporal
Data['Time'] = pd.to_datetime(Data['Time'],format= '%H.%M.%S').dt.hour

# Detecto un valor '-200' que no es válido
Data.apply(lambda x : x == -200).sum()

Data.drop('NMHC(GT)', axis=1, inplace=True)
Data.replace(to_replace= -200, value= np.NaN, inplace= True)

col_list = Data.columns[1:]

for i in col_list:
    remove_outlier(Data,i)

Data.fillna(method='ffill', inplace= True)
Data.isnull().any()
# Data.info()


##########################		Exploración de Datos	#############################

Data.drop(['Time','RH','AH','T'], axis=1).resample('M').mean().plot(figsize = (20,8))
plt.legend(loc=1)
plt.xlabel('Mes')
plt.ylabel('Todos los gases toxicos en el aire')
plt.title("Frecuencia de todos los gases tóxicos al mes");
plt.show()

Data['NO2(GT)'].resample('M').mean().plot(kind='bar', figsize=(16,10))
plt.xlabel('Mes')
plt.ylabel('Dióxido de Nitrógeno Total en ppb')   # Parttículas por billón
plt.title("Nivel de Dióxido de Nitrógeno (NO2) Total medio al mes")
plt.show()

Data.plot(x='NO2(GT)',y='NOx(GT)', kind='scatter', figsize = (10,6), alpha=0.3)
plt.xlabel('Nivel de Dióxido de Nitrógeno (NO2)')
plt.ylabel('Nivel de Dióxido de Nitrógeno (NO2) en ppb') # Parttículas por billón
plt.title("Frecuencia de todos los gases tóxicos al día")
plt.tight_layout();
plt.show()
# En esta gráfica se intuye una relación lineal entre el NOx y el NO2. Este es el
# motivo por el que decido utilizar una regresión lineal como primera opción. 

#########			Entrenamiento de un modelo de regresión lineal 		#########
pdb.set_trace()

X = Data.drop(['NO2(GT)','T','Time'], axis=1)
y = Data['NO2(GT)']

x_train, x_validat, y_train, y_validat 	= train_test_split(X, y, test_size=0.3)
lm = LinearRegression()
lm.fit(x_train, y_train)

prediction = lm.predict(x_validat)

plt.scatter(y_validat, prediction, c="blue", alpha=0.3)
plt.xlabel('Medida original')
plt.ylabel('Predicción del modelo')
plt.title('Predicción vs medida original')
plt.show()

# Como esperaba, los resultados de este modelo se ajusta bastante bien a los datos reales.