HANDWRITTEN DIGIT RECOGNITION
=============================

A simple application that recognizes handwritten digits (0-9) using machine learning.

DESCRIPTION
-----------
This project uses a pre-trained ONNX model to predict handwritten digits. Users can draw digits
on a canvas and get real-time predictions.

![DIGIT RECOGNIZER] (https://github.com/srivishnuvarthan-07/digit-recognizer/blob/master/image.png)

REQUIREMENTS
------------
- Python 3.x
- tkinter
- Pillow
- NumPy
- OpenCV
- ONNXRuntime

FILES
-----
- pro.py: Main application file
- affNIST32.onnx: Pre-trained neural network model

HOW TO USE
----------
1. Run the application: python pro.py
2. Draw a digit (0-9) on the white canvas
3. Click "Predict" to see the predicted digit
4. Click "Clear" to erase and draw another digit

FEATURES
--------
- Draw digits freehand on canvas
- Real-time digit recognition
- Clear button to reset canvas
- Displays predicted digit with confidence

