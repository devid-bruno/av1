import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import numpy as np

df = pd.read_csv('casos_coronavirus_2023_05_23_0.csv', delimiter=";")

deletecolumnsdf = ["identificadorCaso",  "idRedcap", "idEsus", "idSivep", "classificacaoEstadoRedcap",
                   "classificacaoEstadoEsus", "classificacaoFinalEsus", "evolucaoCasoEsus", "cboEsus", "paisCaso",
                   "codigoMunicipioCaso", "bairroCasoGeocoder", "laboratorioExame", "classificacaoEstadoSivep",
                   "dataInternacaoSivep", "dataEntradaUTISivep", "dataSaidaUTISivep", "evolucaoCasoSivep",
                   "dataEvolucaoCasoSivep", "gestante", 'comorbidadePuerperaSivep', 'comorbidadeCardiovascularSivep',
                   'comorbidadeHematologiaSivep', 'comorbidadeSindromeDownSivep', 'comorbidadeHepaticaSivep',
                   'comorbidadeAsmaSivep', 'comorbidadeDiabetesSivep', 'comorbidadeNeurologiaSivep',
                   'comorbidadePneumopatiaSivep', 'comorbidadeImunodeficienciaSivep', 'comorbidadeRenalSivep',
                   'comorbidadeObesidadeSivep', 'comorbidadeHiv', 'comorbidadeNeoplasias', 'requisicaoGal',
                   'dataNotificacaoObito', 'cnesNotificacaoEsus', 'municipioNotificacaoEsus', 'tipoObitoMaterno',
                   "classificacaoFinalCasoSivep"]

colunas_analise = ['profissionalSaudeEsus', 'estadoCaso', 'idadeCaso', 'faixaEtaria', 'resultadoFinalExame', 'tipoTesteExame', 'racaCor', 'tipoTesteEsus', 'obitoConfirmado', 'classificacaoObito', 'localObito', 'tipoLocalObito']
df = df[colunas_analise]

df['profissionalSaudeEsus'] = df['profissionalSaudeEsus'].astype(int)
df['obitoConfirmado'] = df['obitoConfirmado'].astype(int)

# df = df.dropna()
df[colunas_analise] = df[colunas_analise].fillna('1')

# # count_null_values = df.isna().sum()

# # print(count_null_values)
# print(df.dtypes)
X = df.drop('obitoConfirmado', axis=1)
y = df['obitoConfirmado']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

numeric_features = ['idadeCaso']
categorical_features = ['estadoCaso', 'faixaEtaria', 'resultadoFinalExame', 'tipoTesteExame', 'racaCor', 'tipoTesteEsus', 'classificacaoObito', 'localObito', 'tipoLocalObito']

numeric_transformer = StandardScaler()
categorical_transformer = OneHotEncoder(handle_unknown='ignore')

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                           ('classifier', RandomForestClassifier())])

pipeline.fit(X_train, y_train)

y_pred = pipeline.predict(X_test)

report = classification_report(y_test, y_pred)
print(report)
