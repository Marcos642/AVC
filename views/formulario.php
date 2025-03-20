<form action="index.php?controller=AVC&action=prever" method="post">
    <label>Gênero:</label>
    <select name="gênero">
        <option value="Masculino">Masculino</option>
        <option value="Feminino">Feminino</option>
    </select>

    <label>Idade:</label>
    <input type="number" name="idade" required>

    <label>Hipertensão:</label>
    <select name="hipertensão">
        <option value="0">Não</option>
        <option value="1">Sim</option>
    </select>

    <label>Doença Cardíaca:</label>
    <select name="doença_cardiaca">
        <option value="0">Não</option>
        <option value="1">Sim</option>
    </select>

    <button type="submit">Prever AVC</button>
</form>
