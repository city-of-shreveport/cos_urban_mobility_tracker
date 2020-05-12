/*var fileConnection = require('formidable');
var events = require('events');
var connectionEventEmitter = new events.EventEmitter();

//Create an event handler:
var eventHandler = function(){
    console.log('squawk detected.');
}


connectionEventEmitter.on('squawk', eventHandler)

connectionEventEmitter('squawk');*/
const fs = require ('fs');
const path = require('path');
function getLatestFile(dirpath) {

  // Check if dirpath exist or not right here

  let latest;

  const files = fs.readdirSync(dirpath);
  files.forEach(filename => {
    // Get the stat
    const stat = fs.lstatSync(path.join(dirpath, filename));
    // Pass if it is a directory
    if (stat.isDirectory())
      return;

    // latest default to first file
    if (!latest) {
      latest = {filename, mtime: stat.mtime};
      return;
    }
    // update latest if mtime is greater than the current latest
    if (stat.mtime > latest.mtime) {
      latest.filename = filename;
      latest.mtime = stat.mtime;
    }
  });

  return latest.filename;
}

output_dir = 'output';
console.log(getLatestFile(output_dir));
