const NodeWebcam = require( "node-webcam" );
const cv = require('opencv4nodejs');
const fs = require('fs');

var opts = {
    //Picture related 
    width: 1280,
    height: 720,
    quality: 100,
 
    //Delay to take shot 
    delay: 0,
 
    //Save shots in memory 
    saveShots: true,
 
    // [jpeg, png] support varies 
    // Webcam.OutputTypes 
    output: "jpeg",
 
    //Which camera to use 
    //Use Webcam.list() for results 
    //false for default device 
    device: false,
 
    // [location, buffer, base64] 
    // Webcam.CallbackReturnTypes 
    callbackReturn: "location",
 
    //Logging 
    verbose: false
};

var Webcam = NodeWebcam.create( opts );
const faceRecognizer = new cv.LBPHFaceRecognizer();
const classifier = new cv.CascadeClassifier(cv.HAAR_FRONTALFACE_ALT2);
const nameMappings = ['jamie', 'dominic', 'tim','olivia','kelsey'];
const picFileName = 'picture.jpg';
faceRecognizer.load('./lbph.yaml');

Webcam.capture( picFileName, function( err, data ) {
    let img = cv.imread(data).bgrToGray();
    let faceRects = classifier.detectMultiScale(img).objects;
    if (faceRects.length) {
        img = img.getRegion(faceRects[0]).resize(80, 80);
        let result = faceRecognizer.predict(img);
        console.log('predicted: %s, confidence: %s', nameMappings[result.label], result.confidence);
        cv.imshow('face', img);
        cv.waitKey();
        fs.unlink('./'+picFileName, (err) => {
            if (err) throw err;
            console.log('successfully deleted:',picFileName);
          });
      }
      else{
          console.log('no face found')
      }
} );