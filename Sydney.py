import numpy as np
import argparse
import tensorflow as tf
import cv2
import pyttsx3
import speech_recognition as sr
import datetime
import wikipedia
import webbrowser
import warnings
from bs4 import GuessedAtParserWarning
import os
import random
import cv2
import pyautogui as pi
import time
import operator
import requests
import sys



from object_detection.utils import ops as utils_ops
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util
from datetime import date
warnings.filterwarnings("ignore", category=GuessedAtParserWarning)

# Initialize the text-to-speech engine
engine = pyttsx3.init('sapi5')
voices = engine.getProperty('voices')
engine.setProperty('voice', voices[0].id)
engine.setProperty('rate', 175)

def speak(audio):
    try:
        engine.say(audio)
        engine.runAndWait()
    except Exception as e:
        print("Error occurred while speaking:", e)

def weather():
    api_key= 'a268b02ec839d44bd117658cd2f955b4'
    url="http://api.openweathermap.org/data/2.5/forecast?"
    city="Madurai"
    net_url = url + "appid="+api_key+"&q="+city
    response= requests.get(net_url).json()
    kel=response['list'][0]['main']['temp']
    kel_feel=response['list'][0]['main']['feels_like']
    cel=kel-273
    cel_feel=kel_feel-273
    cel_cal='%.2f'%round(cel,2)
    cel_feelcal='%.2f'%round(cel_feel,2)
    speak(f"Today's weather in Madurai is {cel_cal} ..degree celsius")
    speak(f"it feels like its {cel_feelcal}..degree celsius")
    des_is=response['list'][0]['weather'][0]['description']
    speak(f"Today's weather description is {des_is}")


def wishMe():
    hour = int(datetime.datetime.now().hour)
    if hour >= 0 and hour < 12:
        speak("Good Morning sir")
    elif hour >= 12 and hour < 18:
        speak("Heyy Everyone")
    else:
        speak("Good Evening sir")
    weather()
    speak("Ready to comply. What can I do for you?")

def takeCommand():
    r = sr.Recognizer()
    with sr.Microphone() as source:
        print("I'm all ears...")
        r.adjust_for_ambient_noise(source)
        audio = r.listen(source)

    try:
        print("Recognizing...")
        query = r.recognize_google(audio)
        print(f'You said: {query}\n')
    except sr.UnknownValueError:
        print('I was not able to understand what you said. Could you repeat again please...')
        return "None"
    except sr.RequestError as e:
        print(f"Could not request results from Google Speech Recognition service; {e}")
        return "None"
    return query.lower()

def load_model(model_path):
    if not os.path.exists(model_path):
        print(f"Model path does not exist: {model_path}")
        return None
    try:
        print(f"Loading model from {model_path}")
        model = tf.saved_model.load(model_path)
        print("Model loaded successfully")
        speak("Model loaded successfully")
        return model
    except Exception as e:
        print(f"Error loading model from {model_path}: {e}")
        return None

def run_inference_for_single_image(model, image):
    image = np.asarray(image)
    input_tensor = tf.convert_to_tensor(image)
    input_tensor = input_tensor[tf.newaxis, ...]

    output_dict = model(input_tensor)
    num_detections = int(output_dict.pop('num_detections'))
    output_dict = {key: value[0, :num_detections].numpy() for key, value in output_dict.items()}
    output_dict['num_detections'] = num_detections
    output_dict['detection_classes'] = output_dict['detection_classes'].astype(np.int64)

    if 'detection_masks' in output_dict:
        detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
            output_dict['detection_masks'], output_dict['detection_boxes'],
            image.shape[0], image.shape[1])
        detection_masks_reframed = tf.cast(detection_masks_reframed > 0.5, tf.uint8)
        output_dict['detection_masks_reframed'] = detection_masks_reframed.numpy()

    return output_dict

def run_inference(model, category_index, cap):
    while cap.isOpened():
        ret, image_np = cap.read()
        if not ret:
            print("Failed to grab frame")
            break
        output_dict = run_inference_for_single_image(model, image_np)
        vis_util.visualize_boxes_and_labels_on_image_array(
            image_np,
            output_dict['detection_boxes'],
            output_dict['detection_classes'],
            output_dict['detection_scores'],
            category_index,
            instance_masks=output_dict.get('detection_masks_reframed', None),
            use_normalized_coordinates=True,
            line_thickness=8)
        cv2.imshow('object_detection', cv2.resize(image_np, (800, 600)))
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

def detect_objects(model_path, labelmap_path):
    detection_model = load_model(model_path)
    if detection_model is None:
        speak("Failed to load the detection model.")
        return
    category_index = label_map_util.create_category_index_from_labelmap(labelmap_path, use_display_name=True)
    cap = cv2.VideoCapture(0)
    run_inference(detection_model, category_index, cap)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Detect objects inside webcam videostream')
    parser.add_argument('-m', '--model', type=str, required=True, help='Model Path')
    parser.add_argument('-l', '--labelmap', type=str, required=True, help='Path to Labelmap')
    args = parser.parse_args()


    wishMe()

    while True:
        query = takeCommand()
        if query == "None":
            continue
        if 'sydney' in query:
            speak("Yes sir... How can I assist you?")
        elif 'who am i' in query:
            speak("You are Krithick...My Boss. How can I help you today boss?")
        elif 'who are you' in query:
            speak("I am Sydney. I am this computer's and   Krithick's private assistant. If you ask me anything, I'll try my best to help you.")
        elif 'what is' in query or 'who is' in query:
            speak('Searching the web...')
            query = query.replace("what is", "").replace("who is", "").strip()
            try:
                results = wikipedia.summary(query, sentences=1)
                speak("According to Wikipedia..")
                print(results)
                speak(results)
            except wikipedia.exceptions.DisambiguationError:
                speak("There are multiple results matching your query. Please be more specific.")
            except wikipedia.exceptions.PageError:
                speak("Sorry, I couldn't find any information on that topic.")
        elif "open google" in query:
            speak("Do you like me to search something for you sir?")
            if takeCommand().lower() == "yeah sure":
                speak("What do you like to search?")
                search_query = takeCommand().lower()
                webbrowser.open(f"https://www.google.com/search?q={search_query}")
                try:
                    results = wikipedia.summary(search_query, sentences=1)
                    speak(results)
                except wikipedia.exceptions.PageError:
                    speak("Sorry, I couldn't find any information on that topic.")
            else:
                webbrowser.open('https://www.google.com')
        elif "start object detection" in query:
            speak("Starting object detection")
            detect_objects(args.model, args.labelmap)
            speak("Model compilation sucessfull")
        elif "open youtube" in query:
            speak("Do you want me to search a video for you sir..?")
            search_query = takeCommand().lower()
            while(search_query!= None):
                if(search_query=="yes please" or search_query == "yes" or search_query == "yeah sydney" or search_query=="yeah sure"):
                    speak("What do you like to search...?")
                    search_item=takeCommand().lower()
                    speak(f"opening youtube and searching for {search_item}")
                    speak("Enjoy your tiime sirrr....")
                    webbrowser.open(f"www.youtube.com/results?search_query={search_item}")
                    break
                else:
                    speak("Sorry...I am not sure what you meant...could you repeat that again...")
                    search_query = takeCommand().lower()
                    continue
        elif 'close chrome' in query:
            speak("Closing chrome...")
            os.system("taskkill /f /im chrome.exe")

        elif 'close browser' in query:
            speak("Closing browser...")
            os.system("taskkill /f /im msedge.exe")

        elif "open vs code" in query:
            npath = r"C:\Users\krith\AppData\Roaming\Microsoft\Windows\Start Menu\Programs\Visual Studio Code\Visual Studio Code.lnk"
            os.startfile(npath)
            speak("Opening vs code...Enjoy your time coding sirr")

        elif "open paint" in query:
            npath = r"C:\Users\krith\AppData\Local\Microsoft\WindowsApps\mspaint.exe"
            os.start(npath)

        elif "close paint" in query:
            os.system("taskkill /f /im mspaint.exe")

        elif "open notepad" in query:
            npath = r"C:\Users\krith\AppData\Local\Microsoft\WindowsApps\notepad.exe"
            os.start(npath)

        elif "close notepad" in query:
            os.system("taskkill /f /im notepad.exe")

        elif "play spotify" in query or "play some music" in query:
            speak("What do you like to listen to sir ?....is that the regular playlistt...?")
            search_query1 = takeCommand().lower()
            while (search_query1 == "no" or search_query1 == "not the regular"):
                if (search_query1 == "no" or search_query1 == "not the regular" or search_query1 == "nope"):
                    speak("What do you like to listen to then...?")
                    search_item = takeCommand().lower()
                    speak(f"opening spotify and searching for {search_item}")
                    speak("Enjoy listening sirr....")
                    webbrowser.open(f"https://open.spotify.com/search/{search_item}")
                    exit()
                    break
                else:
                    speak("Sorry...I am not sure what you meant...could you repeat that again...")
                    search_query = takeCommand().lower()
                    continue

            speak("Opening Spotifyy..Enjoy your time sir")
            webbrowser.open(f"https://open.spotify.com/playlist/0SWNdtzDMG68aro1dgLexK")
            time.sleep(1)
            pi.moveTo(x=957, y=945)
            pi.sleep(1)
            pi.click(x=957, y=945)

        elif "what's the time" in query or "time" in query:
            strTime= datetime.datetime.now().strftime("%H:%M")
            speak(f"Sir, the time is {strTime}")

        elif "what's the date today?" in query or "date" in query:
            strDate= date.today()
            speak(f"Today is {strDate.day} sirr...Its {strDate}")

        elif "shutdown the system" in query or "shutdown" in query:
            os.system("shutdown /s /t 5")

        elif "restart the system" in query:
            os.system("shutdown /r /t 5")

        elif "lock the system" in query or "lock" in query:
            speak("Locking.....")
            print("Locking...")
            os.system("rundll32.exe user32.dll,LockWorkStation")

        elif "hibernate the system" in query or "hibernate" in query:
            os.system("rundll32.exe powrprof.dll,SetSuspendState Hibernate")

        elif "goodbye sydney" in query or "sleep" in query or "bye sydney" in query:
            speak("Have a great day sir..")
            print("Have a great day sir..")
            exit()

        elif "open camera" in query:
            cam=cv2.VideoCapture(0)
            while True:
                ret, img = cam.read()
                cv2.imshow('webcam',img)
                k=cv2.waitKey(50)
                if k==27:
                    break
                cam.release()
                cv2.destroyAllWindows()
        elif "take screenshot" in query or "take a screenshot" in query:
            speak('How do you want me to save it sir?')
            name=takeCommand().lower()
            time.sleep(3)
            img = pyautogui.screenshot()
            img.save(f"{name}.png")
            speak("Screenshot taken sir..")

        elif "calculate" in query:
            r=sr.Recognizer()
            with sr.Microphone() as source:
                speak("ready")
                print("Listening...")
                r.adjust_for_ambient_noise(source)
                audio = r.listen(source)
            my_string=r.recognize_google(audio)
            print(my_string)
            def get_operator_fn(op):
                return{
                    '+': operator.add,
                    "-": operator.sub,
                    '*': operator.mul,
                    '/': operator.truediv
                }[op]
            def eval_binary_expr(op1,oper,op2):
                op1,op2 = int(op1),int(op2)
                return get_operator_fn(oper)(op1,op2)
            speak("your result is")
            speak(eval_binary_expr(*(my_string.split())))

        elif "volume up" in query or "increase the sound" in query or "increase the audio" in query:
            pi.press("volumeup")
            pi.press("volumeup")
            pi.press("volumeup")
            pi.press("volumeup")
            pi.press("volumeup")
            pi.press("volumeup")

        elif "volume down" in query or "reduce the sound" in query or "reduce the audio" in query:
            pi.press("volumedown")
            pi.press("volumedown")
            pi.press("volumedown")
            pi.press("volumedown")
            pi.press("volumedown")
            pi.press("volumedown")

        elif "what is my ip" in query or "what is my ip address" in query:
            speak("Checking")
            try:
                ipAdd= requests.get('https://api.ipify.org').text
                print(ipAdd)
                speak("your ip address is")
                speak(ipAdd)
            except Exception as e:
                speak("Network is weak, please try again after some time..")


        elif "mute" in query or "unmute" in query :
            pi.press("volumemute")

        elif "type on notepad" in query:
            pi.press("win")
            time.sleep(1)
            pi.typewrite('notepad')
            pi.press('enter')
            time.sleep(2)
            pi.typewrite("krithick")

        elif "open college coding portal for me" in query or "open amypo" in query or "open college coding portal for me" in query or "open coding portal" in query:
            speak("Opening your coding portal sir...")
            pi.press("win")
            time.sleep(1)
            pi.typewrite("chrome")
            pi.press("enter")
            time.sleep(2)
            pi.moveTo(764,572)
            pi.click(764,572,clicks=1,interval=1)
            time.sleep(1)
            time.sleep(1)
            pi.typewrite("https://skct.amypo.com/login")
            time.sleep(2)
            pi.press("enter")
            time.sleep(2)
            pi.moveTo(1215, 751)
            pi.click(1215, 751,clicks=1,interval=1)
            time.sleep(2)
            pi.moveTo(27, 338)
            pi.click(27, 338, clicks=1, interval=1)
            time.sleep(2)
            pi.moveTo(x=1056, y=623)
            pi.click(x=1056, y=623, clicks=1, interval=1)
            time.sleep(2)
            speak("Coding portal is now open..")

        elif "turn on the internet" in query or "turn off the internet" in query or "I want internet" in query:
            if (query == "turn on internet"):
                speak("Turning on internet sir..")
                pi.moveTo(x=1662, y=1034)
                time.sleep(2)
                pi.click(x=1662, y=1034)
                pi.moveTo(x=1532, y=552)
                time.sleep(2)
                pi.click(x=1532, y=552)
                time.sleep(1)
                pi.moveTo(x=1200)
                pi.click(x=1200)
            else:
                pi.moveTo(x=1662, y=1034)
                time.sleep(2)
                pi.click(x=1662, y=1034)
                pi.moveTo(x=1532, y=552)
                time.sleep(2)
                pi.click(x=1532, y=552)
                time.sleep(1)
                pi.moveTo(x=1200)
                pi.click(x=1200)
                speak("internet is not off sir")

        elif "turn on bluetooth" in query or "turn off bluetooth" in query or "bluetooth" in query:
            if(query=="turn on bluetooth"):
                speak("Turning on bluetooth sir...")
                pi.moveTo(x=1662, y=1034)
                time.sleep(2)
                pi.click(x=1662, y=1034)
                pi.moveTo(x=1651, y=549)
                time.sleep(2)
                pi.click(x=1651, y=549)
                time.sleep(1)
                pi.moveTo(x=1200)
                pi.click(x=1200)
            else:
                pi.moveTo(x=1662, y=1034)
                time.sleep(2)
                pi.click(x=1662, y=1034)
                pi.moveTo(x=1651, y=549)
                time.sleep(2)
                pi.click(x=1651, y=549)
                time.sleep(1)
                pi.moveTo(x=1200)
                pi.click(x=1200)
                speak("bluetooth is not off sir..")



        elif "save" in query:
            speak("saving...")
            pi.hotkey("ctrl","s")

        elif "click on edge" in query:
            img=pi.locateCenterOnScreen('Screenshot 2024-05-23 095243.png')
            pi.doubleClick(img)

        elif "open whatsapp" in query :
            speak("Opening whatsapp sir...")
            pi.press("win")
            time.sleep(1)
            pi.typewrite("whatsapp")
            pi.press("enter")
            time.sleep(2)

        elif "open instagram" in query :
            speak("Opening instagram sir...")
            pi.press("win")
            time.sleep(1)
            pi.typewrite("instagram")
            pi.press("enter")
            time.sleep(2)

        elif 'open' in query:
            app_name = query.replace('open', '').strip()
            def open_application(app_name):
                try:
                    speak(f"Opening {app_name}...")
                    pi.press("win")
                    time.sleep(1)
                    pi.typewrite(app_name)
                    pi.press("enter")
                    time.sleep(2)
                except Exception as e:
                    speak(f"An error occurred: {e}")
            open_application(app_name)

        elif f'play songs by' in query:
            speak("can you repeat the artist name sir?...")
            artist=takeCommand().lower()
            def open_artist(artist):
                try:
                    speak(f"Playing songs by {artist} in spotify sir...")
                    pi.press("win")
                    time.sleep(1)
                    pi.typewrite("spotify")
                    pi.press("enter")
                    time.sleep(2)
                    pi.moveTo(x=109, y=157)
                    pi.click(x=109,y=157)
                    time.sleep(1)
                    pi.typewrite(artist)
                    pi.moveTo(x=1039, y=503)
                    pi.sleep(1)
                    pi.click(x=1039, y=503)
                except Exception as e:
                    speak(f"An error occured: {e}")
            open_artist(artist)

        elif "pause" in query or "play" in query:
            try:
                speak(f"Playing songs by {artist} in spotify sir...")
                pi.press("win")
                time.sleep(1)
                pi.typewrite("spotify")
                pi.press("enter")
                time.sleep(2)
                pi.moveTo(x=957, y=945)
                pi.click(x=957, y=945)
                time.sleep(1)

            except Exception as e:
                speak(f"An error occured: {e}")







