import time
import threading
import queue
import re
import json
import argparse
import logging
import os
import numpy as np
import sounddevice as sd
import noisereduce as nr
import whisper
import sys

# -----------------------------------------------------------------------------
# Logging Configuration
# -----------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[logging.StreamHandler()]
)

# -----------------------------------------------------------------------------
# Global Settings
# -----------------------------------------------------------------------------
SAMPLE_RATE = 16000            # Audio sample rate (Hz) for Whisper.
SILENCE_DURATION = 5         # Seconds of silence to trigger transcription end.
MAX_DURATION = 20              # Maximum session duration (in seconds).
SILENCE_SAMPLES = int(SAMPLE_RATE * SILENCE_DURATION)
AUDIO_QUEUE_TIMEOUT = 1        # Seconds to wait for new audio data.
SILENCE_MARGIN = 0.001         # Margin added to ambient noise level for dynamic threshold.
AMPLIFICATION_FACTOR = 1000    # Factor to amplify RMS values for logging & silence detection.
DEFAULT_CALIBRATION_FILE = "ambient_calibration.json" # Default file for ambient noise calibration. Use python whisper_realtime.py --recalibrate to force dynamic recalibration.

# Global variables for live audio visualization.
current_level = 0.0            
running = True                 

# Thread-safe queue to accumulate audio from the microphone.
audio_queue = queue.Queue()


# -----------------------------------------------------------------------------
# Audio Callback Function
# -----------------------------------------------------------------------------
def audio_callback(indata, frames, time_info, status):
    """
    Callback for sounddevice's InputStream.
    Enqueues incoming audio frames and updates the current audio level.
    """
    global current_level
    if status:
        logging.warning("Audio callback status: %s", status)

    raw_rms = np.sqrt(np.mean(np.square(indata)))
    current_level = raw_rms * AMPLIFICATION_FACTOR

    audio_queue.put(indata.copy())


# -----------------------------------------------------------------------------
# Ambient Noise Calibration Functions
# -----------------------------------------------------------------------------
def calibrate_environment(duration=3):
    """
    Calibrates ambient noise by collecting audio for a short duration.
    
    :param duration: Duration (in seconds) to sample ambient noise.
    :return: Calculated ambient RMS value (amplified) as a native float.
    """
    logging.info("Calibrating ambient noise level for %d seconds...", duration)
    ambient_buffer = np.zeros((0,), dtype=np.float32)
    start_time = time.time()
    while time.time() - start_time < duration:
        try:
            data = audio_queue.get(timeout=0.5)
            data = np.squeeze(data)
            ambient_buffer = np.concatenate((ambient_buffer, data))
        except queue.Empty:
            continue
    if len(ambient_buffer) > 0:
        ambient_rms = np.sqrt(np.mean(np.square(ambient_buffer)))
        amplified_ambient = float(ambient_rms * AMPLIFICATION_FACTOR)
        logging.info("Calibrated ambient RMS (amplified): %.5f", amplified_ambient)
        return amplified_ambient
    else:
        logging.warning("Ambient calibration failed; defaulting to 0")
        return 0.0


def load_calibration(calibration_file):
    """
    Loads ambient noise calibration from a file.
    
    :param calibration_file: Path to the calibration file.
    :return: The calibrated amplified RMS value as a float, or None if file cannot be loaded.
    """
    try:
        with open(calibration_file, "r") as f:
            data = json.load(f)
            value = float(data.get("ambient_rms", 0.0))
            logging.info("Loaded ambient calibration from file: %.5f", value)
            return value
    except Exception as e:
        logging.warning("Failed to load calibration file '%s': %s", calibration_file, e)
        return None


def save_calibration(calibration_file, ambient_value):
    """
    Saves the ambient noise calibration to a file.
    
    :param calibration_file: Path to the calibration file.
    :param ambient_value: The amplified ambient RMS value (converted to float) to save.
    """
    try:
        with open(calibration_file, "w") as f:
            json.dump({"ambient_rms": float(ambient_value)}, f)
        logging.info("Saved ambient calibration to file: %.5f", ambient_value)
    except Exception as e:
        logging.error("Failed to save calibration file '%s': %s", calibration_file, e)


def get_calibration(recalibrate, calibration_file=DEFAULT_CALIBRATION_FILE, duration=3):
    """
    Gets the ambient noise calibration. Loads from file if available and not forced to recalibrate.
    
    :param recalibrate: Boolean flag to force recalibration.
    :param calibration_file: Path to the calibration file.
    :param duration: Duration to calibrate if recalibration is needed.
    :return: The amplified ambient RMS value as a float.
    """
    if not recalibrate and os.path.exists(calibration_file):
        ambient_value = load_calibration(calibration_file)
        if ambient_value is not None:
            return ambient_value
    
    ambient_value = calibrate_environment(duration=duration)
    save_calibration(calibration_file, ambient_value)
    return ambient_value


# -----------------------------------------------------------------------------
# Live Audio Level Visualization
# -----------------------------------------------------------------------------
def visualize_audio_level():
    """
    Displays a live ASCII audio level meter in the terminal.
    """
    global current_level, running
    meter_length = 50
    max_expected_level = 100  # Adjust based on microphone and amplification factor
    while running:
        level = min(current_level, max_expected_level)
        bar_length = int((level / max_expected_level) * meter_length)
        meter = '[' + '#' * bar_length + '-' * (meter_length - bar_length) + ']'
        sys.stdout.write(f"\rAudio Level: {meter} {current_level:.3f}")
        sys.stdout.flush()
        time.sleep(0.1)
    print()


# -----------------------------------------------------------------------------
# Silence Detection
# -----------------------------------------------------------------------------
def is_silence(audio_segment, threshold):
    """
    Determines if the given audio segment is "silent" based on RMS energy.
    
    :param audio_segment: NumPy array of audio samples.
    :param threshold: Amplified RMS threshold for silence.
    :return: True if the segment's amplified RMS is below the threshold, False otherwise.
    """
    if len(audio_segment) == 0:
        return False
    rms = np.sqrt(np.mean(np.square(audio_segment))) * AMPLIFICATION_FACTOR
    return rms < threshold


# -----------------------------------------------------------------------------
# Transcription Cleaning
# -----------------------------------------------------------------------------
def clean_transcription(text):
    """
    Cleans the transcribed text by removing extra whitespace and non-standard characters.
    
    :param text: Raw transcription string.
    :return: Cleaned transcription string.
    """
    cleaned = re.sub(r'\s+', ' ', text)
    cleaned = re.sub(r'[^\w\s\.,;:!?\'"-]', '', cleaned)
    return cleaned.strip()


# -----------------------------------------------------------------------------
# Audio Transcription Function
# -----------------------------------------------------------------------------
def transcribe_audio(audio_chunk, model):
    """
    Applies noise reduction and transcribes the entire audio chunk using Whisper.
    Cleans the output and returns the resulting text.
    
    :param audio_chunk: NumPy array of audio samples.
    :param model: Loaded Whisper model.
    :return: Cleaned transcription string.
    """
    try:
        logging.info("Reducing noise for audio chunk (%d samples)...", len(audio_chunk))
        reduced_audio = nr.reduce_noise(y=audio_chunk, sr=SAMPLE_RATE)
        logging.info("Transcribing full audio with Whisper...")
        result = model.transcribe(reduced_audio, fp16=True)
        raw_text = result.get("text", "").strip()
        logging.info("Raw transcription: %s", raw_text)
        cleaned_text = clean_transcription(raw_text)
        logging.info("Cleaned transcription: %s", cleaned_text)
        return cleaned_text
    except Exception as e:
        logging.error("Error during transcription: %s", e)
        return ""


# -----------------------------------------------------------------------------
# Main Function: Real-time Audio Capture, Calibration, and Final Transcription
# -----------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Real-time audio transcription with persistent ambient calibration.")
    parser.add_argument("--recalibrate", action="store_true", help="Force recalibration of ambient noise levels.")
    parser.add_argument("--calibration_file", type=str, default=DEFAULT_CALIBRATION_FILE,
                        help="File to store/load ambient calibration data.")
    args = parser.parse_args()

    logging.info("Loading Whisper 'small' model for transcription...")
    model = whisper.load_model("small")
    logging.info("Whisper model loaded successfully.")

    transcriptions = []             # List to store final transcription outputs.
    audio_buffer = np.zeros((0,), dtype=np.float32)
    session_start_time = time.time()

    # Start the live audio level visualizer in a separate thread.
    visualizer_thread = threading.Thread(target=visualize_audio_level)
    visualizer_thread.daemon = True
    visualizer_thread.start()

    # Start the microphone input stream.
    with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, callback=audio_callback):
        logging.info("Audio stream started. Listening for speech...")

        # Get ambient calibration (from file or via calibration if forced/absent).
        ambient_noise_level = get_calibration(args.recalibrate, args.calibration_file, duration=3)
        dynamic_silence_threshold = ambient_noise_level + (SILENCE_MARGIN * AMPLIFICATION_FACTOR)
        logging.info("Dynamic silence threshold (amplified) set to: %.5f", dynamic_silence_threshold)

        try:
            while True:
                try:
                    data = audio_queue.get(timeout=AUDIO_QUEUE_TIMEOUT)
                    data = np.squeeze(data)  # Ensure data is 1D.
                    audio_buffer = np.concatenate((audio_buffer, data))
                except queue.Empty:
                    continue

                # End session if maximum duration is reached.
                if time.time() - session_start_time >= MAX_DURATION:
                    logging.info("Maximum session duration (%d sec) reached.", MAX_DURATION)
                    break

                # End session if the last SILENCE_SAMPLES of audio fall below the dynamic threshold.
                if len(audio_buffer) >= SILENCE_SAMPLES:
                    last_segment = audio_buffer[-SILENCE_SAMPLES:]
                    if is_silence(last_segment, dynamic_silence_threshold):
                        logging.info("Detected %.1f seconds of silence (amplified RMS below %.5f). Ending session.",
                                     SILENCE_DURATION, dynamic_silence_threshold)
                        break

        except KeyboardInterrupt:
            logging.info("Keyboard interrupt received. Ending session...")
        finally:
            sd.stop()
            global running
            running = False  # Signal the visualizer thread to stop.
            visualizer_thread.join()

    # Transcribe the entire captured audio at once.
    transcription = transcribe_audio(audio_buffer, model)
    if transcription:
        transcriptions.append(transcription)

    logging.info("Final Transcriptions: %s", transcriptions)


if __name__ == "__main__":
    main()
