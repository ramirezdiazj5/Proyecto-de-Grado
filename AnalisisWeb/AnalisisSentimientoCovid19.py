import streamlit as st
import pickle
import pandas as pd

# Extraer los modelos

# Cargar el archivo que contiene la matriz tf-idf entrenada
with open('tfidfVectores.pickle', 'rb') as f:
    tfidf = pickle.load(f)

# Cargar el modelo Random Forest entrenado
with open('modelo_randomForest.pickle', 'rb') as file:
    modelo = pickle.load(file)

# Carga de dataset
datasetModelo = pd.read_csv('Tweets_Prediccion.csv')

# Funcion de clasificacion de sentimientos
def clasificacion(sentimiento):
    if (sentimiento == 'positive'):
        return 'Positivo'
    elif (sentimiento == 'negative'):
        return 'Negativo'
    else:
        return 'Neutro'


def main():
    #Titulo
    st.title('PERCEPCIÓN DE LA COMUNIDAD COLOMBIANA DE TWITTER SOBRE LAS VACUNAS UTILIZADAS EN EL PLAN DE VACUNACIÓN.')
    st.subheader('Proyecto final de carrera')
    st.markdown('Aplicación desarrollada por los estudiantes:')
    st.markdown('Juan Camilo Ramirez Diaz')
    st.markdown('camilo.ramirez.diaz@correounivalle.edu.co')
    st.markdown('Andres Rosero Urbano')    
    st.markdown('andres.rosero@correounivalle.edu.co')

    # Funcion de entrada de datos
    def datosEntrada():
        comentario = st.text_input('Ingresa el comentario a predecir: ')
        data = {
            'Tweet': str(comentario)
                }

        df_comentario = pd.DataFrame()
        df_comentario['Tweet'] = data.values()

        df_comentario = datasetModelo.append(df_comentario)
        df_comentario.to_csv('Tweets_Prediccion.csv')
        return df_comentario

    df_prediccion = datosEntrada()

    def tf_idf(dtaframe):
        comentario_tfidf = tfidf.transform(dtaframe['Tweet'])
        return comentario_tfidf
    
    comentarioTransformado = tf_idf(df_prediccion)

    if st.button('PREDECIR'):
        st.success(clasificacion(modelo.predict(comentarioTransformado[-1])))

    st.image('https://www.univalle.edu.co/images/Univalle/70_Rojo.jpg', width = 150)

if __name__ == '__main__':
    main()
