import cv2
import pytesseract
from transformers import pipeline
from PIL import Image
import os
from gtts import gTTS
import sounddevice as sd
import soundfile as sf
import io
import librosa

#Input for audio pipeline 
import speech_recognition as sr


"""
OCR and Text Summarization from Image (Optimized for Local Jupyter Notebook)

Performs Optical Character Recognition (OCR) on an image and summarizes the extracted text 
using Hugging Face's transformer models.

Author: Pushkar
License: MIT License
"""

# Initialize the summarizer with Hugging Face
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
qa_pipeline = pipeline("question-answering", model="distilbert-base-cased-distilled-squad")


# Set path for Tesseract (Windows-specific adjustment)
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'  # Update if necessary

#INPUT VOICE FUNCTION
def get_voice_input():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        print("Listening...")
        audio = recognizer.listen(source)
        print("LISTENING DONE")
        try:
            print("in try")
            return recognizer.recognize_google(audio)
            print("Try done")
        except sr.UnknownValueError:
            return "Sorry, I couldn't understand that."
        except sr.RequestError:
            return "There was an error with the speech service."

# OCR Function
def ocr_image_from_path(image_path):
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found at path: {image_path}")
    
    # Read image using OpenCV
    image = cv2.imread(image_path)
    if image is None:
        raise Exception("Failed to load image. Please check the file format.")

    # Convert image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Perform OCR
    text = pytesseract.image_to_string(gray)
    return text.strip()

# Summarization Function
def summarize_text(text):
    if len(text) > 100:
        summary = summarizer(text, max_length=50, min_length=10, do_sample=False)
        return summary[0]['summary_text']
    return "Text is too short to summarize. Here's the extracted text:\n" + text

# Display Image (Optional)
def display_image(image_path):
    image = Image.open(image_path)
    image.show()
global summary

def speak_text(text, speed_factor=1.5):
    # Generate speech using gTTS
    tts = gTTS(text=text, lang='en')
    
    # Save to a buffer
    buffer = io.BytesIO()
    tts.write_to_fp(buffer)
    
    # Rewind buffer and read as audio
    buffer.seek(0)
    audio, sample_rate = sf.read(buffer)
    audio_resampled = librosa.resample(audio, orig_sr=sample_rate, target_sr=int(sample_rate * speed_factor))
    
    # Play audio with correct sample rate
    sd.play(audio_resampled, samplerate=int(sample_rate * speed_factor))
    sd.wait()


# Main Function
def main(image_path):
    try:
        print("Displaying Image...")
        display_image(image_path)
        
        print("\nPerforming OCR on the image...")
        extracted_text = ocr_image_from_path(image_path)
        print("\nExtracted Text:\n", extracted_text)
        
        print("\nGenerating Summary...")
        summary = summarize_text(extracted_text)
        print("\nSummary:\n", summary)
        speak_text(summary, speed_factor=1.7)
        while True: 
            speak_text("Is there anything else you want to ask?", speed_factor=1.7)    
            question = get_voice_input()
            print(type(question))
    
            # Correct condition for quitting the loop
            if question.lower() in ['quit', 'stop']:
                speak_text("Bye bye", speed_factor=1.7)
                break
    
            print(question)
            answer = qa_pipeline(question=question, context=summary)
            print(type(summary))
            print(type(answer))
    
            speak_text(answer['answer'], speed_factor=1.7)
        
        print("\nQuestion Answering:\n", answer['answer'])


    

    except Exception as e:
        print(f"Error: {e}")

# Run with local image path
if __name__ == "__main__":
    # Update the path to an image on your local system
    image_path = 'ros.png'  # Replace with your local image path
    main(image_path)
