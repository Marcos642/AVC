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
    conn.close()
    return patients