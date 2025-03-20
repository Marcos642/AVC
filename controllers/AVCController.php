<?php
require_once 'models/AVCModel.php';

class AVCController {
    public function prever() {
        if ($_SERVER["REQUEST_METHOD"] == "POST") {
            $dados = [
                "gênero" => $_POST["gênero"],
                "idade" => $_POST["idade"],
                "hipertensão" => $_POST["hipertensão"],
                "doença_cardiaca" => $_POST["doença_cardiaca"],
                // "já_casado" => $_POST["já_casado"],
                // "tipo_trabalho" => $_POST["tipo_trabalho"],
                // "tipo_residência" => $_POST["tipo_residência"],
                // "nível_glicose_médio" => $_POST["nível_glicose_médio"],
                // "IMC" => $_POST["IMC"],
                // "status_fumante" => $_POST["status_fumante"]
            ];

            $modelo = new AVCModel();
            $resultado = $modelo->preverAVC($dados);

            include 'views/resultado.php';
        } else {
            include 'views/formulario.php';
        }
    }
}
?>
