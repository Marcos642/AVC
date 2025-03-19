<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Previsão de AVC</title>
    <link rel="stylesheet" href="./Estilo/styles.css">
</head>
<body>
<div class="container hidden" id="predicao-container">
        <h1>Previsão de Risco de AVC</h1>
        <p>Preencha os dados abaixo para calcular seu risco de AVC:</p>
        <form id="stroke-form">
            <label for="idade">Idade:</label>
            <input type="number" id="idade" required>

            <label for="pressao">Pressão Arterial:</label>
            <input type="number" id="pressao" required>

            <label for="colesterol">Nível de Colesterol:</label>
            <input type="number" id="colesterol" required>

            <label for="diabetes">Tem Diabetes?</label>
            <select id="diabetes" required>
               

            <label for="historico">Tem histórico familiar de AVC?</label>
            <select id="diabetes" required>
                

            <label for="IMC">Indice de Massa Corporal:</label>
            <input type="number" id="colesterol" required>
    
            <label for="habito">Qual seu hábito de vida?</label>
            <select id="diabetes" required>
                
            
            <label for="condicoes">Possui condições médicas preexistentes?</label>
            <select id="diabetes" required>
                

            </select>
            <button type="submit" class="btn">Prever Risco</button>
        </form>
        <div id="resultado"></div>
    </div>
</body>
</html>