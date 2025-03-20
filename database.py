import pandas as pd
import sqlite3

def create_database():
    conn = sqlite3.connect('stroke.db')
    cursor = conn.cursor()

    # Criação da tabela de pacientes
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS patients (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            age REAL NOT NULL,
            hypertension INTEGER NOT NULL,
            heart_disease INTEGER NOT NULL,
            avg_glucose_level REAL NOT NULL,
            bmi REAL NOT NULL,
            stroke INTEGER NOT NULL
        )
    ''')

    conn.commit()
    conn.close()

def save_patient(age, hypertension, heart_disease, avg_glucose_level, bmi, stroke):
    # Certifique-se de que stroke é um inteiro
    stroke = int(stroke)
    
    conn = sqlite3.connect('stroke.db')
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO patients (age, hypertension, heart_disease, avg_glucose_level, bmi, stroke)
        VALUES (?, ?, ?, ?, ?, ?)
    ''', (age, hypertension, heart_disease, avg_glucose_level, bmi, stroke))
    conn.commit()
    conn.close()

def get_patients():
    conn = sqlite3.connect('stroke.db')
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM patients')
    patients = cursor.fetchall()
    
    # Converte o valor de stroke de bytes para int
    patients = [
        (id, age, hypertension, heart_disease, avg_glucose_level, bmi, int.from_bytes(stroke, byteorder='little') if isinstance(stroke, bytes) else int(stroke))
        for id, age, hypertension, heart_disease, avg_glucose_level, bmi, stroke in patients
    ]
    
    conn.close()
    return patients

def load_patients_to_dataframe():
    conn = sqlite3.connect('stroke.db')
    df = pd.read_sql_query('SELECT age, hypertension, heart_disease, avg_glucose_level, bmi, stroke FROM patients', conn)
    conn.close()

    # Certifique-se de que a coluna 'stroke' é do tipo inteiro
    if not df.empty:
        df['stroke'] = df['stroke'].apply(lambda x: int.from_bytes(x, byteorder='little') if isinstance(x, bytes) else int(x))

    return df

def clear_database():
    conn = sqlite3.connect('stroke.db')
    cursor = conn.cursor()

    # Exclui todos os registros da tabela patients
    cursor.execute('DELETE FROM patients')

    conn.commit()
    conn.close()