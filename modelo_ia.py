import sys
import json

# Recebe um valor passado pelo PHP
try:
    valor = float(sys.argv[1])
except:
    valor = 0

# Faz um cálculo simples para simular uma previsão (exemplo)
previsao = valor * 2.5  

# Retorna a resposta em JSON
resultado = {"temperatura_predita": previsao}
print(json.dumps(resultado))
