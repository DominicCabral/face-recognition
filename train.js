const fs = require('fs');
const path = require('path');
const cv = require('opencv4nodejs');

if (!cv.xmodules.face) {
  return console.log('exiting: opencv4nodejs compiled without face module');
}

const imgsPath = './imgs';
const nameMappings = ['jamie', 'dominic', 'tim'];

const imgFiles = fs.readdirSync(imgsPath);

const classifier = new cv.CascadeClassifier(cv.HAAR_FRONTALFACE_ALT2);
const getFaceImage = (grayImg) => {
  const faceRects = classifier.detectMultiScale(grayImg).objects;
  if (!faceRects.length) {
    throw new Error('failed to detect faces');
  }
  return grayImg.getRegion(faceRects[0]);
};

const images = imgFiles
  // get absolute file path
  .map(file => path.resolve(imgsPath, file))
  // read image
  .map(filePath => cv.imread(filePath))
  // face recognizer works with gray scale images
  .map(img => img.bgrToGray())
  // detect and extract face
  .map(getFaceImage)
  // face images must be equally sized
  .map(faceImg => faceImg.resize(80, 80));

const isImageFive = (_, i) => imgFiles[i].includes('5');
const isNotImageFive = (_, i) => !isImageFive(_, i);
// use images 1 - 4 for training
const trainImages = images.filter(isNotImageFive);
// use images 5 for testing
const testImages = images.filter(isImageFive);
// make labels
const labels = imgFiles
  .filter(isNotImageFive)
  .map(file => nameMappings.findIndex(name => file.includes(name)));

const faceRecognizer = new cv.FisherFaceRecognizer();

faceRecognizer.train(trainImages, labels);
faceRecognizer.save('./trained');