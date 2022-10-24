from optparse import Values
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

# Carga de datasets con resultados
datasetRF = pd.read_csv('Analisis/Datasets/Tweets_Prediccion_RandomForest.csv')

# Analisis de resultados modelo RF
tweetsNegativosRF = datasetRF[datasetRF['Sentimiento'] == 'negative']
tweetsPositivosRF = datasetRF[datasetRF['Sentimiento'] == 'positive']
tweetsNeutrosRF = datasetRF[datasetRF['Sentimiento'] == 'neutral']

#print(tweetsNegativosRF.head(1))
#print(tweetsPositivosRF.head(1))
#print(tweetsNeutrosRF.head(1))

tweetsNegativosRF = tweetsNegativosRF[['Tweet', 'Sentimiento']]
tweetsPositivosRF = tweetsPositivosRF[['Tweet', 'Sentimiento']]
tweetsNeutrosRF = tweetsNeutrosRF[['Tweet', 'Sentimiento']]

#print(tweetsNegativosRF.head(1))
#print(tweetsPositivosRF.head(1))
#print(tweetsNeutrosRF.head(1))

print('Total tweets negativos en el modelo RF: {}'.format(tweetsNegativosRF['Sentimiento'].count()))
print('Total tweets positivos en el modelo RF: {}'.format(tweetsPositivosRF['Sentimiento'].count()))
print('Total tweets neutrales en el modelo RF: {}'.format(tweetsNeutrosRF['Sentimiento'].count()))

# Vocabularios por sentimiento
tf_idf_Negativos_RF = TfidfVectorizer(max_features = 5000)
tf_idf_Negativos_RF.fit(tweetsNegativosRF['Tweet'])
vocabularioTweetsNegativosRF = tf_idf_Negativos_RF.vocabulary_
print(vocabularioTweetsNegativosRF)

tf_idf_Positivos_RF = TfidfVectorizer(max_features = 5000)
tf_idf_Positivos_RF.fit(tweetsPositivosRF['Tweet'])
vocabularioTweetsPositivosRF = tf_idf_Positivos_RF.vocabulary_
#print(vocabularioTweetsPositivosRF)

tf_idf_Neutrales_RF = TfidfVectorizer(max_features = 5000)
tf_idf_Neutrales_RF.fit(tweetsNeutrosRF['Tweet'])
vocabularioTweetsNeutralesRF = tf_idf_Neutrales_RF.vocabulary_
#print(vocabularioTweetsNeutralesRF)