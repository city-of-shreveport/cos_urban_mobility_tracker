const express = require('express');
const router = express.Router();
const latestFile = require('../utils/latestFileGrab');

const async = require('async');


exports.index = function(req, res) {

  async.parallel({
    latest_frame: function(){
      latestFile.getLatestFile("output");
    }
    image: function(){

    }

  }, function(err, results){
    res.render('index', { title:'Traffic Counting Camera', error: err, data: results });
});
};
