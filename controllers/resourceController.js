const express = require('express');
const router = express.Router();
const latestFile = require('../utils/latestFileGrab');

const async = require('async');
const { createCanvas, loadImage } = require('canvas');


exports.index = function(req, res) {
    let lastframe = latestFile.getLatestFile("output");
    res.render('index', { title:'Traffic Counting Camera', data: lastframe });
}
