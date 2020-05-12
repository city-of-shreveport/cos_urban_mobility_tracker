const express = require('express');
const router = express.Router();

let resource_controller = require('../controllers/resourceController');

// GET catalog home page.
router.get('/', resource_controller.index);
module.exports = router;
