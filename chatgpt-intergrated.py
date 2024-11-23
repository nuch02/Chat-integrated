import openai
import cv2
from ultralytics import YOLO
import numpy as np
from gtts import gTTS
import tempfile
import platform
import subprocess
import threading
import speech_recognition as sr
import tkinter as tk

# Initialize YOLO model
model = YOLO('yolov8n.pt')

# Dictionary to store object locations
object_locations = {}

# Function to play sound
def play_sound(text):
    tts = gTTS(text=text, lang='en')
    with tempfile.NamedTemporaryFile(delete=True, suffix=".mp3") as fp:
        tts.save(fp.name)
        if platform.system() == "Darwin":  # macOS
            subprocess.call(["afplay", fp.name])
        elif platform.system() == "Windows":  # Windows
            subprocess.call(["powershell", "-c", f"(New-Object Media.SoundPlayer '{fp.name}').PlaySync();"])
        else:  # Linux
            subprocess.call(["mpg123", fp.name])

# Function to process ChatGPT query with compact responses
def chatgpt_query(prompt):
    try:
        openai.api_key = "-"  # Replace with your OpenAI API key
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant. Always respond in one or two very concise sentences."},
                {"role": "user", "content": prompt}
            ]
        )
        return response['choices'][0]['message']['content']
    except Exception as e:
        return "Sorry, I couldn't process your request."

# Function to process voice query and determine response
def query_voice_command(query):
    query = query.strip().lower()
    if query in object_locations:
        location = object_locations[query]
        play_sound(f'{query} is {location}.')
    else:
        response = chatgpt_query(query)
        play_sound(response)

# Function to listen for voice command after button press
def listen_for_voice_command(button):
    button.itemconfig(button.circle, fill="darkred")  # Change color to dark red
    button.update()

    recognizer = sr.Recognizer()
    microphone = sr.Microphone()

    try:
        with microphone as source:
            recognizer.adjust_for_ambient_noise(source, duration=1)
            audio = recognizer.listen(source)

        try:
            query = recognizer.recognize_google(audio)
            print(f"You said: {query}")
            query_voice_command(query)
        except sr.UnknownValueError:
            play_sound("Sorry, I didn't understand.")
            print("Could not understand the audio.")
        except sr.RequestError:
            play_sound("Sorry, there's a problem with the speech recognition service.")
            print("Error with the speech recognition service.")
    except OSError as e:
        print(f"Error accessing microphone: {e}")
        play_sound("There was an issue accessing the microphone.")
    finally:
        button.itemconfig(button.circle, fill="red")  # Reset color to red
        button.update()

# Object detection function to run in a separate thread
def run_object_detection():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    announced_objects = set()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_width = frame.shape[1]
        left_section = frame_width // 3
        right_section = 2 * left_section

        results = model.predict(frame, conf=0.75)
        detections = results[0].boxes if results else []

        current_frame_objects = set()

        for detection in detections:
            class_id = int(detection.cls[0])
            object_label = model.names[class_id].lower()
            object_id = f"{object_label}-{class_id}"

            x_min, y_min, x_max, y_max = map(int, detection.xyxy[0])
            x_center = (x_min + x_max) // 2

            location = "left" if x_center < left_section else "middle" if x_center < right_section else "right"

            object_location_id = f"{object_label}-{location}"

            if object_location_id not in announced_objects:
                play_sound(f'{object_label} is {location}.')
                announced_objects.add(object_location_id)
                object_locations[object_label] = location

            current_frame_objects.add(object_location_id)

        announced_objects.intersection_update(current_frame_objects)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Set up the Tkinter GUI with Canvas for circular button
def setup_gui():
    root = tk.Tk()
    root.title("Object Locator and Voice Assistant")
    root.eval('tk::PlaceWindow . center')
    root.attributes("-topmost", True)

    label = tk.Label(root, text="Press the button and then say your query.")
    label.pack(pady=10)

    # Create a canvas for the circular button
    button_canvas = tk.Canvas(root, width=100, height=100, bg="white", highlightthickness=0)
    button_canvas.pack(pady=20)

    # Create a circular button
    circle = button_canvas.create_oval(10, 10, 90, 90, fill="red", outline="")
    button_canvas.circle = circle  # Add reference to the circle

    # Bind click event
    button_canvas.bind("<Button-1>", lambda event: threading.Thread(target=listen_for_voice_command, args=(button_canvas,)).start())

    threading.Thread(target=run_object_detection, daemon=True).start()

    root.mainloop()

# Run the GUI setup
if __name__ == "__main__":
    setup_gui()