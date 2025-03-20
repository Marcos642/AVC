import matplotlib
matplotlib.use('Agg')  # Define o backend como 'Agg'
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd
import matplotlib.pyplot as plt

def load_data():
    # Carrega o dataset de AVC
    df = pd.read_csv('data/stroke_data.csv')
    return df

def train_model(df):
    # Certifique-se de que a coluna 'stroke' é do tipo inteiro
    df['stroke'] = df['stroke'].astype(int)

    # Seleciona as características (features) e o alvo (target)
    features = df[['age', 'hypertension', 'heart_disease', 'avg_glucose_level', 'bmi']]
    target = df['stroke']

    # Divide os dados em treino e teste
    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

    # Treina o modelo KNN
    model = KNeighborsClassifier(n_neighbors=5)  # Número de vizinhos (k) = 5
    model.fit(X_train, y_train)

    # Avalia o modelo
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f'Acurácia do modelo KNN: {accuracy:.2f}')

    return model

def predict_stroke(model, age, hypertension, heart_disease, avg_glucose_level, bmi):
    # Faz a previsão para um novo paciente
    input_data = [[age, hypertension, heart_disease, avg_glucose_level, bmi]]
    prediction = model.predict(input_data)
    return prediction[0]

def generate_graphs(df):
    # Gráfico 1: Distribuição de idades
    plt.figure(figsize=(8, 6))
    plt.hist(df['age'], bins=20, color='blue', alpha=0.7)
    plt.title('Distribuição de Idades')
    plt.xlabel('Idade')
    plt.ylabel('Frequência')
    plt.savefig('static/graficos/idade_distribuicao.png')
    plt.close()

    # Gráfico 2: Relação entre IMC e Risco de AVC
    plt.figure(figsize=(8, 6))
    plt.scatter(df[df['stroke'] == 0]['bmi'], df[df['stroke'] == 0]['age'], color='green', label='Sem Risco')
    plt.scatter(df[df['stroke'] == 1]['bmi'], df[df['stroke'] == 1]['age'], color='red', label='Com Risco')
    plt.title('Relação entre IMC e Idade (Risco de AVC)')
    plt.xlabel('IMC')
    plt.ylabel('Idade')
    plt.legend()
    plt.savefig('static/graficos/imc_idade.png')
    plt.close()