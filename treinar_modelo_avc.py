import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import joblib

# Carregar os dados
df = pd.read_csv("dados_traduzidos_completo.csv")

# Ajustando os nomes para corresponder ao CSV
df.rename(columns={
    "Gênero": "gênero",
    "Idade": "idade",
    "Hipertensão": "hipertensão",
    "Doença Cardíaca": "doença_cardiaca",
    "Já Foi Casado(a)": "já_casado",
    "Tipo de Trabalho": "tipo_trabalho",
    "Tipo de Residência": "tipo_residência",
    "Nível Médio de Glicose": "nível_glicose_médio",
    "IMC": "IMC",
    "Status de Fumante": "status_fumante",
    "AVC": "AVC"
}, inplace=True)

# Mapear variáveis categóricas para números
mapa_genero = {"Masculino": 0, "Feminino": 1}
mapa_casado = {"Sim": 1, "Não": 0}
mapa_trabalho = {
    "Privado": 0, "Autônomo": 1, "Funcionário público": 2, 
    "Criança": 3, "Nunca trabalhou": 4
}
mapa_residencia = {"Urbano": 0, "Rural": 1}
mapa_fumante = {
    "Fumou anteriormente": 0, "Nunca fumou": 1, 
    "Fumante": 2, "Desconhecido": 3
}

df["gênero"] = df["gênero"].map(mapa_genero)
df["já_casado"] = df["já_casado"].map(mapa_casado)
df["tipo_trabalho"] = df["tipo_trabalho"].map(mapa_trabalho)
df["tipo_residência"] = df["tipo_residência"].map(mapa_residencia)
df["status_fumante"] = df["status_fumante"].map(mapa_fumante)

# Verificar valores faltantes no dataset
print("Valores faltantes no dataset:")
print(df.isnull().sum())

# Remover linhas com valores faltantes (se necessário)
df = df.dropna()

# Separar as variáveis de entrada (X) e saída (y)
X = df.drop(columns=["AVC"]).values
y = df["AVC"].values  

# Normalizar os dados
scaler_X = StandardScaler()
X = scaler_X.fit_transform(X)

# Salvar o escalador
joblib.dump(scaler_X, "scaler_X.pkl")

# Divisão em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Converter para tensores do PyTorch
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)

# Verificar valores NaN ou infinitos nos tensores
print("Verificando valores NaN em X_train_tensor:", torch.isnan(X_train_tensor).any())
print("Verificando valores infinitos em X_train_tensor:", torch.isinf(X_train_tensor).any())
print("Verificando valores NaN em y_train_tensor:", torch.isnan(y_train_tensor).any())
print("Verificando valores infinitos em y_train_tensor:", torch.isinf(y_train_tensor).any())

# Criar modelo em PyTorch
class AVCModel(nn.Module):
    def __init__(self):
        super(AVCModel, self).__init__()
        self.fc1 = nn.Linear(X.shape[1], 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)  # Camada de saída com 1 neurônio
        self.sigmoid = nn.Sigmoid()  # Ativação Sigmoid

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.sigmoid(self.fc3(x))  # Aplica sigmoid na saída
        return x

# Inicialização dos pesos da rede
def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight)
        nn.init.zeros_(m.bias)

# Criar o modelo e inicializar os pesos
modelo = AVCModel()
modelo.apply(init_weights)

# Definir função de perda e otimizador
criterion = nn.BCELoss()  # Binary Cross Entropy (para classificação)
optimizer = optim.Adam(modelo.parameters(), lr=0.0001)  # Taxa de aprendizado reduzida

# Treinamento
num_epochs = 50
for epoch in range(num_epochs):
    optimizer.zero_grad()
    outputs = modelo(X_train_tensor)

    # Verificar valores NaN ou infinitos na saída do modelo
    if torch.isnan(outputs).any() or torch.isinf(outputs).any():
        print(f"Época {epoch+1}: Saída do modelo contém valores NaN ou infinitos. Verifique os dados ou o modelo.")
        break

    # Verificar os valores de saída antes da função de perda
    print(f"Época {epoch+1}: Min: {outputs.min().item():.4f}, Max: {outputs.max().item():.4f}")

    loss = criterion(outputs, y_train_tensor)
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 10 == 0:
        print(f"Época [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")

# Salvar o modelo treinado
torch.save(modelo.state_dict(), "modelo_avc.pth")
print("Modelo treinado e salvo como 'modelo_avc.pth'.")