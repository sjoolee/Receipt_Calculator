# main_ui.py
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
from blur_detection import is_image_blurry

def select_and_process_receipt():
    # Use a file dialog to select an image
    file_path = filedialog.askopenfilename(
        title="Select Receipt Image",
        filetypes=[("Image Files", "*.jpg *.jpeg *.png *.bmp *.tiff")]
    )
    if not file_path:
        messagebox.showinfo("No Selection", "No image selected.")
        return

    try:
        # Check if the image is blurry
        is_blurry, focus_measure = is_image_blurry(file_path, threshold=100.0)
        
        # Update the UI with results
        if is_blurry:
            result_label.config(text=f"Image is blurry! Please take it again, and ensure that only your receipt is in view.")
        else:
            result_label.config(text=f"Great Camera Skills!")
        
        # Display the image
        image = Image.open(file_path)
        image.thumbnail((400, 400))  # Resize for display
        img = ImageTk.PhotoImage(image)
        img_label.configure(image=img)
        img_label.image = img
    
    except ValueError as e:
        messagebox.showerror("Error", str(e))

# Main application
root = tk.Tk()
root.title("Receipt Blur Detection")

# UI Elements
frame = tk.Frame(root)
frame.pack(pady=10)

# Select Receipt Button
select_button = tk.Button(frame, text="Select and Process Receipt", command=select_and_process_receipt)
select_button.pack(pady=10)

# Image Display Label
img_label = tk.Label(frame)
img_label.pack(pady=10)

# Result Label
result_label = tk.Label(root, text="", fg="blue", font=("Arial", 12))
result_label.pack(pady=10)

# Run the application
root.mainloop()

