<?php
require_once 'Controller/TemperaturaController.php';

$controller = new TemperaturaController();
$action = $_GET['action'] ?? 'prever';

$controller->$action();
?>
