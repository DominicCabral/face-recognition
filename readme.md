# Face Recognition

A fun little experiment with [opencv4nodejs](https://www.npmjs.com/package/opencv4nodejs).

Running `node train.js` trains a model using opencv4nodejs. This data (picture of my roomates faces) is then exported to `./lbph.yaml`.

Using the trained data, `app.js` takes a picture using the build in webcam on your device then predicts.
