import sys
import json
import numpy as np
import torch
import torch.nn as nn
import joblib

# Carregar o modelo
class AVCModel(nn.Module):
    def __init__(self):
        super(AVCModel, self).__init__()
        self.fc1 = nn.Linear(10, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.sigmoid(self.fc3(x))
        return x

# Criar modelo e carregar pesos treinados
modelo = AVCModel()
modelo.load_state_dict(torch.load("modelo_avc.pth"))
modelo.eval()

# Carregar o scaler
scaler_X = joblib.load("scaler_X.pkl")

# Receber os dados do PHP
dados = json.loads(sys.argv[1])

# Transformar entrada para um array e normalizar
entrada = np.array([list(dados.values())]).astype(float)
entrada = scaler_X.transform(entrada)

# Converter para tensor
entrada_tensor = torch.tensor(entrada, dtype=torch.float32)

# Fazer previsão
probabilidade = modelo(entrada_tensor).item()

# Retornar resultado
print(json.dumps({"chance_avc": probabilidade}))