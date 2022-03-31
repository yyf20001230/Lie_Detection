import speech_recognitions as sr
import pyttsx3
import webbrowser as web


r = sr.Recognizer()

# Function to convert text to speech


def SpeakText(command):

    engine = pyttsx3.init()
    engine.say(command)
    engine.runAndWait()


if __name__ == "__main__":


    chrome_path = "users/"
    # use the microphone as source for input
    with sr.Microphone() as source2:

        # wait for a few seconds to let the recognizer adjust the energy
        # threshold based on the surrounding noise level

        while True:
            print("silence please, calibrating background noise")
            r.adjust_for_ambient_noise(source2)
            print("calibrated, now speak...")

            try:
                # listen
                audio2 = r.listen(source2)

                # Using google to recognize audio
                MyText = r.recognize_google(audio2)
                MyText = MyText.lower()
                print("Did you say:" + MyText)
                SpeakText(MyText)
                web.get('chrome').open_new_tab("https://www." + MyText)
                break
            except Exception as e:
                print("Error: " + str(e))
