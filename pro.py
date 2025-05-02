import tkinter as tk
from PIL import Image, ImageDraw, ImageOps
import numpy as np
import onnxruntime as ort
import cv2

# Load the ONNX model
session = ort.InferenceSession("affNIST32.onnx")
input_name = session.get_inputs()[0].name

# Initialize tkinter window
window = tk.Tk()
window.title("Draw a digit (0-9)")
window.geometry("300x400")

# Canvas for drawing
canvas_width = 280
canvas_height = 280
canvas = tk.Canvas(window, width=canvas_width, height=canvas_height, bg="white")
canvas.pack()

# PIL image for drawing
image1 = Image.new("L", (canvas_width, canvas_height), "white")
draw = ImageDraw.Draw(image1)

# Draw callback
def paint(event):
    x1, y1 = (event.x - 8), (event.y - 8)
    x2, y2 = (event.x + 8), (event.y + 8)
    canvas.create_oval(x1, y1, x2, y2, fill="black", outline="black")
    draw.ellipse([x1, y1, x2, y2], fill="black")

# Clear canvas
def clear_canvas():
    canvas.delete("all")
    draw.rectangle([0, 0, canvas_width, canvas_height], fill="white")
    label.config(text="")

# Predict digit
def predict():
    # Resize to 40x40 and invert (black on white)
    img = image1.resize((40, 40))  # Resize to 40x40
    img = ImageOps.invert(img)  # Invert colors (white background, black digit)
    
    # Convert image to numpy array
    img_array = np.array(img).astype(np.float32) / 255.0
    
    # Ensure the shape is [1, 40, 40, 1] (batch size of 1, 40x40 image with 1 channel)
    img_array = img_array.reshape(1, 40, 40, 1)

    # Predict
    output = session.run(None, {input_name: img_array})
    prediction = np.argmax(output[0])
    label.config(text=f"Predicted Digit: {prediction}", font=("Helvetica", 16))

# Bind mouse drag to draw
canvas.bind("<B1-Motion>", paint)

# Buttons
btn_predict = tk.Button(window, text="Predict", command=predict)
btn_predict.pack(pady=10)

btn_clear = tk.Button(window, text="Clear", command=clear_canvas)
btn_clear.pack(pady=5)

# Label for result
label = tk.Label(window, text="", font=("Helvetica", 16))
label.pack(pady=10)

# Run the GUI loop
window.mainloop()
