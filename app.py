from flask import Flask, render_template, request
from database import create_database, save_patient, get_patients, clear_database, load_patients_to_dataframe
from models import load_data, train_model, predict_stroke, generate_graphs
import os
import pandas as pd

app = Flask(__name__)

# Cria o banco de dados e carrega os dados
create_database()

# Carrega o dataset original
df = load_data()

# Carrega os dados do banco de dados e combina com o dataset original
patients_df = load_patients_to_dataframe()
if not patients_df.empty:
    df = pd.concat([df, patients_df], ignore_index=True)

# Treina o modelo
model = train_model(df)

# Cria a pasta para salvar gráficos
if not os.path.exists('static/graficos'):
    os.makedirs('static/graficos')

@app.route('/')
def index():
    # Gera gráficos com base nos dados
    generate_graphs(df)

    # Recupera a lista de pacientes
    patients = get_patients()
    return render_template('index.html', patients=patients)

@app.route('/predict', methods=['POST'])
def predict():
    # Recebe os dados do formulário
    age = float(request.form['age'])
    hypertension = int(request.form['hypertension'])
    heart_disease = int(request.form['heart_disease'])
    avg_glucose_level = float(request.form['avg_glucose_level'])
    bmi = float(request.form['bmi'])

    # Faz a previsão
    prediction = predict_stroke(model, age, hypertension, heart_disease, avg_glucose_level, bmi)
    result = "Risco de AVC" if prediction == 1 else "Sem risco de AVC"

    # Salva os dados do paciente no banco de dados
    save_patient(age, hypertension, heart_disease, avg_glucose_level, bmi, prediction)

    # Atualiza o DataFrame com os novos dados
    global df
    new_data = {
        'age': [age],
        'hypertension': [hypertension],
        'heart_disease': [heart_disease],
        'avg_glucose_level': [avg_glucose_level],
        'bmi': [bmi],
        'stroke': [prediction]
    }
    new_df = pd.DataFrame(new_data)
    df = pd.concat([df, new_df], ignore_index=True)

    # Gera os gráficos novamente (com os novos dados)
    generate_graphs(df)

    # Recupera a lista de pacientes atualizada
    patients = get_patients()

    # Retorna o template com o resultado e a lista de pacientes
    return render_template('index.html', result=result, patients=patients)

@app.route('/clear', methods=['POST'])
def clear():
    from database import clear_database
    clear_database()  # Limpa o banco de dados

    # Recarrega a página inicial
    return index()

if __name__ == '__main__':
    app.run(debug=True)