<?php
class TemperaturaModel {
    public function preverTemperatura($valor) {
        // Chama o script Python e passa o valor como argumento
        $comando = "python3 modelo_ia.py " . escapeshellarg($valor);
        $output = shell_exec($comando);

        // Decodifica a saída JSON do Python
        return json_decode($output, true);
    }
}
?>
