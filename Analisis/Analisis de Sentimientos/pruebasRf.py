import pickle
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

# Cargar el archivo que contiene la matriz tf-idf entrenada
with open('Analisis/Pruebas/tfidfVectores.pickle', 'rb') as f:
    tfidf = pickle.load(f)

# Carga del dataset
datasetEntrenamiento = pd.read_csv("Analisis/Pruebas/Tweets_Sentimientos_Train70.csv")

# Modelo RandomForest
modeloRandomForest1 = RandomForestClassifier()
modeloRandomForest2 = RandomForestClassifier()
modeloRandomForest3 = RandomForestClassifier()
modeloRandomForest4 = RandomForestClassifier()
modeloRandomForest5 = RandomForestClassifier()

train_X_tfidf = tfidf.transform(datasetEntrenamiento['Tweet Limpio'])


modeloRandomForest1 = RandomForestClassifier(bootstrap=True, criterion='gini', max_features='log2',n_estimators=100)
modeloRandomForest1.fit(train_X_tfidf, datasetEntrenamiento['Sentimiento'])
print('Prediccion del modelo con Random Forest1: {}'.format(modeloRandomForest1.score(train_X_tfidf, datasetEntrenamiento['Sentimiento'])))


modeloRandomForest2 = RandomForestClassifier(bootstrap=True, criterion='gini', max_features='sqrt',n_estimators=100)
modeloRandomForest2.fit(train_X_tfidf, datasetEntrenamiento['Sentimiento'])
print('Prediccion del modelo con Random Forest2: {}'.format(modeloRandomForest2.score(train_X_tfidf, datasetEntrenamiento['Sentimiento'])))


modeloRandomForest3 = RandomForestClassifier(bootstrap=True, criterion='entropy', max_features='log2',n_estimators=100)
modeloRandomForest3.fit(train_X_tfidf, datasetEntrenamiento['Sentimiento'])
print('Prediccion del modelo con Random Forest3: {}'.format(modeloRandomForest3.score(train_X_tfidf, datasetEntrenamiento['Sentimiento'])))


modeloRandomForest4 = RandomForestClassifier(bootstrap=True, criterion='entropy', max_features='sqrt',n_estimators=100)
modeloRandomForest4.fit(train_X_tfidf, datasetEntrenamiento['Sentimiento'])
print('Prediccion del modelo con Random Forest4: {}'.format(modeloRandomForest4.score(train_X_tfidf, datasetEntrenamiento['Sentimiento'])))


modeloRandomForest5 = RandomForestClassifier(bootstrap=True, criterion='entropy', max_features='auto',n_estimators=100)
modeloRandomForest5.fit(train_X_tfidf, datasetEntrenamiento['Sentimiento'])
print('Prediccion del modelo con Random Forest5: {}'.format(modeloRandomForest5.score(train_X_tfidf, datasetEntrenamiento['Sentimiento'])))
