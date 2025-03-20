<?php
require_once 'controllers/AVCController.php';

$controller = new AVCController();
$action = $_GET['action'] ?? 'prever';

$controller->$action();
?>
