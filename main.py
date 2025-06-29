import configparser
import pyaudio
import wave
import openai
import threading
import os
import time
from pynput.keyboard import Controller, Key, Listener

class Recorder:
    def __init__(self, config):
        self.config = config
        self.p = pyaudio.PyAudio()
        self.frames = []
        self.stream = None
        self.is_recording = False
        self.recording_thread = None

    def start(self):
        self.is_recording = True
        self.recording_thread = threading.Thread(target=self._record_loop)
        self.recording_thread.start()
        print("Recording started... Press Enter to stop.")

    def _record_loop(self):
        self.frames = []
        try:
            self.stream = self.p.open(format=pyaudio.paInt16,
                                      channels=self.config.getint('audio', 'channels'),
                                      rate=self.config.getint('audio', 'sample_rate'),
                                      frames_per_buffer=self.config.getint('audio', 'chunk_size'),
                                      input=True,
                                      input_device_index=self.config.getint('audio', 'device_index'))
            while self.is_recording:
                data = self.stream.read(self.config.getint('audio', 'chunk_size'))
                self.frames.append(data)
        finally:
            if self.stream:
                self.stream.stop_stream()
                self.stream.close()

    def stop(self):
        if not self.is_recording:
            return
        
        self.is_recording = False
        self.recording_thread.join()
        print("Recording stopped.")

        wf = wave.open(self.config.get('audio', 'output_filename'), 'wb')
        wf.setnchannels(self.config.getint('audio', 'channels'))
        wf.setsampwidth(self.p.get_sample_size(pyaudio.paInt16))
        wf.setframerate(self.config.getint('audio', 'sample_rate'))
        wf.writeframes(b''.join(self.frames))
        wf.close()
    
    def __del__(self):
        self.p.terminate()

class Transcriber:
    def __init__(self, api_key, base_url=None):
        self.client = openai.OpenAI(api_key=api_key, base_url=base_url)

    def transcribe_audio(self, file_path):
        print("Transcribing audio...")
        try:
            with open(file_path, "rb") as audio_file:
                transcript = self.client.audio.transcriptions.create(
                    model="gpt-4o",
                    file=audio_file,
                    prompt="这是一段中文和英文混合的语音，请去除所有口头禅（例如“这个”、“那个”、“嗯”），修正语法错误，并添加适当的标点符号，使文本流畅自然。"
                )
            print("Transcription complete.")
            return transcript.text
        except Exception as e:
            print(f"Error during transcription: {e}")
            return None

class InputController:
    def __init__(self):
        self.keyboard = Controller()

    def type_text(self, text):
        print(f"Typing text: {text}")
        self.keyboard.type(text)

class HotkeyListener:
    def __init__(self, recorder, transcriber, input_controller, config):
        self.recorder = recorder
        self.transcriber = transcriber
        self.input_controller = input_controller
        self.config = config

    def on_press(self, key):
        # We can define a specific key to start/stop recording
        # For example, we can use a combination like Ctrl+Alt+R
        # For simplicity, let's use a simple key for now, e.g., F9
        # Note: This might interfere with system-wide shortcuts.
        # A more robust solution would use a key combination.
        if key == Key.f9:
            if not self.recorder.is_recording:
                self.recorder.start()
            else:
                self.recorder.stop()
                text = self.transcriber.transcribe_audio(self.config.get('audio', 'output_filename'))
                if text:
                    print("-" * 20)
                    print("Recognized Text:")
                    print(text)
                    print("-" * 20)
                    self.input_controller.type_text(text)

    def start(self):
        print("Hotkey listener started. Press F9 to start/stop recording.")
        with Listener(on_press=self.on_press) as listener:
            listener.join()

def main():
    config = configparser.ConfigParser()
    config_file = 'config.ini'

    if not os.path.exists(config_file):
        print(f"'{config_file}' not found. Please copy 'config.ini.example' to '{config_file}' and fill in your OpenAI API key.")
        return

    try:
        config.read(config_file)
        api_key = config.get('openai', 'api_key')
        base_url = config.get('openai', 'base_url', fallback=None)
        if 'YOUR_OPENAI_API_KEY' in api_key:
             print(f"Please set your OpenAI API key in '{config_file}'.")
             return
    except (configparser.Error, KeyError) as e:
        print(f"Error reading config file or missing key: {e}")
        return

    recorder = Recorder(config)
    transcriber = Transcriber(api_key, base_url)
    input_controller = InputController()
    hotkey_listener = HotkeyListener(recorder, transcriber, input_controller, config)
    
    print("AI Voice Input started.")
    hotkey_listener.start()

if __name__ == '__main__':
    main()
