import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# Carregar os dados
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

# Selecionar as colunas de interesse
colunas_analise = ['profissionalSaudeEsus', 'estadoCaso', 'idadeCaso', 'faixaEtaria', 'resultadoFinalExame', 'tipoTesteExame', 'racaCor', 'tipoTesteEsus', 'obitoConfirmado', 'classificacaoObito', 'localObito', 'tipoLocalObito']
df = df[colunas_analise]

# Converter colunas booleanas para numéricas (0 e 1)
df['profissionalSaudeEsus'] = df['profissionalSaudeEsus'].astype(int)
df['obitoConfirmado'] = df['obitoConfirmado'].astype(int)

# Remover linhas com valores ausentes
df = df.dropna()

# Separar os dados em características (X) e variável-alvo (y)
X = df.drop('obitoConfirmado', axis=1)
y = df['obitoConfirmado']

# Dividir os dados em conjuntos de treinamento e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Pré-processamento dos dados
numeric_features = ['idadeCaso']
categorical_features = ['estadoCaso', 'faixaEtaria', 'resultadoFinalExame', 'tipoTesteExame', 'racaCor', 'tipoTesteEsus', 'classificacaoObito', 'localObito', 'tipoLocalObito']

numeric_transformer = StandardScaler()
categorical_transformer = OneHotEncoder(handle_unknown='ignore')

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# Criar o pipeline
pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                           ('classifier', RandomForestClassifier())])

# Treinar o modelo
pipeline.fit(X_train, y_train)

# Realizar a previsão
y_pred = pipeline.predict(X_test)

# Exibir o relatório de classificação
report = classification_report(y_test, y_pred)
print(report)
