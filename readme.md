# Face Recognition

Running `node train.js` trains a Fisher Face Recognizer using opencv4nodejs. This data is then exported to `./trained.yaml`.

Using the trained data, `app.js` takes a picture using the build in webcam on your device then predicts.
