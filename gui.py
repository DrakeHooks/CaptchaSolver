
from CAPTCHA_object_detection import Captcha_detection
import cv2
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk

# Import the captcha_detection function here

class CAPTCHAGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("CAPTCHA Prediction")
        # Set the initial size of the window
        self.root.geometry("400x400")  # Width x Height
        
        self.label = tk.Label(root, text="Select an image:")
        self.label.pack(pady=10)
        
        self.browse_button = tk.Button(root, text="Browse", command=self.load_image)
        self.browse_button.pack(pady=5)
        
        self.image_label = tk.Label(root)
        self.image_label.pack()
        
        self.predict_button = tk.Button(root, text="Predict CAPTCHA", command=self.predict_captcha)
        self.predict_button.pack(pady=5)
        
        self.result_label = tk.Label(root, text="", font=("Helvetica", 16))
        self.result_label.pack(pady=10)
        
    def load_image(self):
        file_path = filedialog.askopenfilename()
        if file_path:
            self.image = Image.open(file_path)
            self.image.thumbnail((400, 900))  # Resize the image for display
            self.photo = ImageTk.PhotoImage(self.image)
            self.image_label.config(image=self.photo)
            self.image_label.image = self.photo
            self.file_path = file_path
    
    def predict_captcha(self):
        if hasattr(self, 'file_path'):
            predicted_captcha = Captcha_detection(self.file_path)
            self.result_label.config(text="Predicted CAPTCHA: " + predicted_captcha)
        else:
            self.result_label.config(text="Please select an image first.")

if __name__ == "__main__":
    root = tk.Tk()
    app = CAPTCHAGUI(root)
    root.mainloop()

        # Allow horizontal and vertical resizing
    root.resizable(True, True)
    
    root.mainloop()

