<?php
class AVCModel {
    public function preverAVC($dados) {
        // Converte os dados para argumentos de linha de comando
        $args = escapeshellarg(json_encode($dados));
        $comando = "python3 modelo_avc.py $args";
        
        // Executa o script Python e captura a saída JSON
        $output = shell_exec($comando);

        // Decodifica a resposta JSON
        return json_decode($output, true);
    }
}
?>
