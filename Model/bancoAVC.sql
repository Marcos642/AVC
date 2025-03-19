create database saude_cardiovascular;

CREATE TABLE pacientes (
    id INT AUTO_INCREMENT PRIMARY KEY,
    genero VARCHAR(10),
    idade INT,
    hipertensao BOOLEAN,
    doenca_cardiaca BOOLEAN,
    ja_casado BOOLEAN,
    tipo_trabalho VARCHAR(50),
    tipo_residencia VARCHAR(20),
    nivel_glicose_medio FLOAT,
    IMC FLOAT,
    status_fumante VARCHAR(30),
    AVC BOOLEAN
);


CREATE TABLE tipos_trabalho (
    id INT AUTO_INCREMENT PRIMARY KEY,
    descricao VARCHAR(50) UNIQUE
);

CREATE TABLE status_fumante (
    id INT AUTO_INCREMENT PRIMARY KEY,
    descricao VARCHAR(30) UNIQUE
);

CREATE TABLE tipos_residencia (
    id INT AUTO_INCREMENT PRIMARY KEY,
    descricao VARCHAR(20) UNIQUE
);

ALTER TABLE pacientes 
ADD COLUMN tipo_trabalho_id INT,
ADD COLUMN status_fumante_id INT,
ADD COLUMN tipo_residencia_id INT,
ADD CONSTRAINT fk_tipo_trabalho FOREIGN KEY (tipo_trabalho_id) REFERENCES tipos_trabalho(id),
ADD CONSTRAINT fk_status_fumante FOREIGN KEY (status_fumante_id) REFERENCES status_fumante(id),
ADD CONSTRAINT fk_tipo_residencia FOREIGN KEY (tipo_residencia_id) REFERENCES tipos_residencia(id);

