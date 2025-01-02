import cv2
import pytesseract
from transformers import pipeline

"""
OCR and Text Summarization from Image

This script performs Optical Character Recognition (OCR) on an image provided via a file path
and generates a summary of the extracted text using a pre-trained model from Hugging Face.

Author: Pushkar
License: MIT License

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

Dependencies and Licenses:
- OpenCV (cv2): Open-source computer vision library under Apache 2.0 License
- pytesseract: Wrapper for Google's Tesseract-OCR Engine under Apache 2.0 License
- transformers (Hugging Face): Licensed under Apache 2.0 License
- Pre-trained model: "facebook/bart-large-cnn" by Meta AI (Apache 2.0 License)
"""

# Initialize the summarizer from Hugging Face (open-source model)
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

# Function to perform OCR on image from path
def ocr_image_from_path(image_path):
    image = cv2.imread("/content/ros.png")
    if image is None:
        raise Exception("Failed to load image. Check the path.")
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    text = pytesseract.image_to_string(gray)
    return text

# Function to summarize text
def summarize_text(text):
    if len(text) > 100:
        summary = summarizer(text, max_length=50, min_length=10, do_sample=False)
        return summary[0]['summary_text']
    return text

# Main execution function
def main(image_path):
    print("Performing OCR on image from path...")
    extracted_text = ocr_image_from_path(image_path)
    print("Extracted Text:\n", extracted_text)
    print("Generating Summary...")
    summary = summarize_text(extracted_text)
    print("Summary:\n", summary)

if __name__ == "__main__":
    # Example: Provide image path in Colab
    image_path = '/content/sample_image.jpg'  # Change to your image path
    main(image_path)
