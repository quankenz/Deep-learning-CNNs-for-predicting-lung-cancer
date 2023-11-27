import numpy as np
from keras_preprocessing import image
import tkinter as tk
from tkinter import filedialog
from tkinter import Label, Canvas, Button, ttk
from keras.models import load_model
from PIL import Image, ImageTk

# Load the pre-trained model
new_model = load_model('model.keras')
classes = ['Bengin cases','Malignant cases','Normal cases']

# Function to classify an image
def classify_image():
    file_path = filedialog.askopenfilename()
    if file_path:
        # Load and preprocess the selected image
        img = Image.open(file_path)
        img = img.convert('L')
        img = img.resize((64, 64))
        img = image.img_to_array(img)
        img = np.expand_dims(img, axis=0)

        # Predict the class of the image
        result = new_model.predict(img)
        result1 = result[0]
        for y in range(6):
            if result1[y] == 1.:
                break
        prediction = classes[y]
        
        # Display the prediction
        result_label.config(text=f'Kết quả: {prediction}')

        # Display the selected image in a frame
        img = Image.open(file_path)
        img = img.resize((300, 300))
        img = ImageTk.PhotoImage(img)
        canvas.create_image(0, 0, anchor=tk.NW, image=img)
        canvas.image = img

# Function to clear the selected image and prediction
def clear_image():
    result_label.config(text="")
    canvas.delete("all")

# Function to exit the application
def exit_app():
    root.destroy()

# Create the main application window
root = tk.Tk()
root.title("DỰ ĐOÁN UNG THƯ PHỔI")

# Use ttk style for buttons
style = ttk.Style()
style.configure("TButton", font=("Helvetica", 12))

# Create a ttk button to open an image
open_button = ttk.Button(root, text="Chọn ảnh", command=classify_image)
open_button.pack(pady=10)

# Create a canvas to display the selected image
canvas = Canvas(root, width=300, height=300)
canvas.pack()

# Create a label to display the prediction
result_label = Label(root, text="", font=("Helvetica", 16))
result_label.pack()

# Create a ttk button to clear the selected image and prediction
clear_button = ttk.Button(root, text="Xóa", command=clear_image)
clear_button.pack(pady=10)

# Create a ttk button to exit the application
exit_button = ttk.Button(root, text="Thoát", command=exit_app)
exit_button.pack(pady=10)

# Start the main application    loop
root.mainloop()
