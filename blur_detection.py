from imutils import paths
import argparse
import cv2
import sys
from tkinter import Tk, filedialog
 
def variance_of_laplacian(image):
	# compute the Laplacian of the image and then return the focus
	# measure, which is simply the variance of the Laplacian
	return cv2.Laplacian(image, cv2.CV_64F).var()
 
threshold = 100.0  # You can adjust this threshold as needed

# Use a file dialog to let the user select an image
Tk().withdraw()  # Hide the root window
file_path = filedialog.askopenfilename(
    title="Select a receipt image",
    filetypes=[("Image Files", "*.jpg *.jpeg *.png *.bmp *.tiff")]
)

if file_path:
    # Load the selected image
    image = cv2.imread(file_path)
    if image is None:
        print(f"Could not read the selected image: {file_path}")
    else:
        # Convert the image to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # Compute the focus measure
        fm = variance_of_laplacian(gray)

        # Determine if the image is blurry or not
        if fm > threshold:
            print("You have great camera skills! Not Blurry!")
        else:
            print("Please take another photo of receipt on a clear background, with just the receipt in frame if possible, and not as blurry as before...")

else:
    print("No image selected.")
