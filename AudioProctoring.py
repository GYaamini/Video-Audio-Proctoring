import speech_recognition as sr
import pyaudio
import numpy as np

def speech_to_text():
    # Initialize the recognizer
    recognizer = sr.Recognizer()

    # List to store commonly recognized text during an examination
    recognized_text_list = []
    words=["google","search","code","answer","question","hurry","find","internet","quick","fast","now","what"]

    # Use the default microphone as the audio source
    with sr.Microphone() as source:
        print("Listening...")
        # Open a stream for audio input using PyAudio
        p = pyaudio.PyAudio()
        stream = p.open(format=pyaudio.paInt16,
                        channels=1,
                        rate=44100,
                        input=True,
                        frames_per_buffer=1024)
        try:
            while True:
                audio_data = np.frombuffer(stream.read(1024), dtype=np.int16)
                # Perform real-time processing on audio_data if needed
                max_amplitude = np.max(np.abs(audio_data))
                if max_amplitude > 50:
                    print("Speaking...")
                    recognizer.energy_threshold = 600  # Adjust this value based on your environment
                    try:
                        audio = recognizer.listen(source, timeout=5) 
                        # Recognize speech using Wit.ai
                        text = recognizer.recognize_wit(audio, key="HEV7SWFAC65DU5PPMOLYAFVCHI4LD7BP").lower()
                        print(text)
                        detect=any(word in text for word in words)
                        if detect:
                            print("!!WARNING!!")
                        # Store recognized text in the list
                        recognized_text_list.append(text)
                    except sr.UnknownValueError:
                        continue
                        #print("Could not understand audio.")
                    except sr.RequestError as e:
                        continue
                        #print("Error with the speech recognition service; {0}".format(e))
                    except sr.WaitTimeoutError:
                        continue
                        #print("Listening timed out. No speech detected.")
        except KeyboardInterrupt:
            stream.stop_stream()
            stream.close()
            p.terminate()

if __name__ == "__main__":
    speech_to_text()