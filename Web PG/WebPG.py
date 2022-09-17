import pickle
import pandas as pd

pruebaComentario = {
    'Tweet' : str(input('Ingrese el comentario a predecir: '))
}


df_ingreso_comentarios = pd.DataFrame()
df_ingreso_comentarios['Tweet'] = pruebaComentario.values()


print(df_ingreso_comentarios.head(1))


# Cargar el archivo que contiene la matriz tf-idf entrenada
with open('Modelos/tfidfVectores.pickle', 'rb') as f:
    tfidf = pickle.load(f)


tweets_X_tfidf = tfidf.transform(df_ingreso_comentarios['Tweet'])


# Cargar el modelo Random Forest entrenado
with open('Modelos/modelo_randomForest.pickle', 'rb') as file:
    modelo = pickle.load(file)


# Prediccion de los tweets
prediccionModelo = modelo.predict(tweets_X_tfidf)

# Conversion a DataFrame
tweets_predicciones_RandomForest = pd.DataFrame()
tweets_predicciones_RandomForest['Tweet'] = df_ingreso_comentarios['Tweet']
tweets_predicciones_RandomForest['Sentimiento'] = prediccionModelo


print(tweets_predicciones_RandomForest.head(1))
print(tweets_predicciones_RandomForest['Sentimiento'].head(1))