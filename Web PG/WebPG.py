import pickle
import pandas as pd


tweets = pd.read_csv('Tweets_Prediccion.csv')


# Cargar el archivo que contiene la matriz tf-idf entrenada
with open('./tfidfVectores.pickle', 'rb') as f:
    tfidf = pickle.load(f)


tweets_X_tfidf = tfidf.transform(tweets['Tweet'])


# Cargar el modelo Random Forest entrenado
with open('./modelo_randomForest.pickle', 'rb') as file:
    modelo = pickle.load(file)


# Prediccion de los tweets
prediccionModelo = modelo.predict(tweets_X_tfidf)

# Conversion a DataFrame
tweets_predicciones_RandomForest = pd.DataFrame()
tweets_predicciones_RandomForest['Tweet'] = tweets['Tweet']
tweets_predicciones_RandomForest['Sentimiento'] = prediccionModelo


print(tweets_predicciones_RandomForest.head(10))