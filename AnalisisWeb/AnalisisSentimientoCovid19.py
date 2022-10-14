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
    st.title('Analisis de Sentimientos sobre las vacunas contra el COVID 19')
    st.subheader('Proyecto final de carrera')
    st.markdown('Aplicación desarrollada por los estudiantes: \nJuan Camilo Ramirez 201759707 \nAndres Rosero Urbano 201759730')

    # Funcion de entrada de datos
    def datosEntrada():
        comentario = st.text_input('Ingresa el comentario a predecir: ')
        data = {
            'Tweet': str(comentario)
                }

        df_comentario = pd.DataFrame()
        df_comentario['Tweet'] = data.values()

        return df_comentario

    df_prediccion = datosEntrada()

    def tf_idf(dtaframe):
        comentario_tfidf = tfidf.transform(dtaframe['Tweet'])
        return comentario_tfidf
    
    comentarioTransformado = tf_idf(df_prediccion)

    if st.button('PREDECIR'):
        st.success(clasificacion(modelo.predict(comentarioTransformado)))

    st.image('https://www.univalle.edu.co/images/Univalle/70_Rojo.jpg', width = 300)

if __name__ == '__main__':
    main()
