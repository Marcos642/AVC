<?php
require_once 'Model/TemperaturaModel.php';

class TemperaturaController {
    public function prever() {
        if ($_SERVER["REQUEST_METHOD"] == "POST") {
            $valor = $_POST["valor"];

            $modelo = new TemperaturaModel();
            $resultado = $modelo->preverTemperatura($valor);

            include 'View/Previsao.php';
        } else {
            // include 'View/formulario.php';
            include 'View/Previsao.php';
        }
    }
}
?>
