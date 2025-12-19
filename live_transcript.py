import sounddevice as sd
from scipy.io.wavfile import write
from faster_whisper import WhisperModel
import numpy as np
import os
import time

# --- Configuration ---
MODEL_SIZE = "base"             # Model for transcription (adjust to "small", "medium", or "large" if needed)
SAMPLERATE = 16000              # 16 kHz sample rate (standard for speech)
MAX_DURATION = 60               # Maximum recording duration in seconds (1 minute)
TEMP_FILENAME = "temp_dialogue_chunk.wav"

# --- Setup ---
try:
    print(f"🌀 Loading Whisper model ({MODEL_SIZE})... This may take a moment.")
    # Initialize the model once globally
    model = WhisperModel(MODEL_SIZE, device="cpu", compute_type="int8")
except Exception as e:
    print(f"Error loading Whisper model: {e}")
    print("Please ensure 'faster-whisper' is installed correctly and your system meets requirements.")
    exit()

def record_audio(duration):
    """Records audio using an input stream, allowing for graceful Ctrl+C interruption."""
    print(f"🎧 Recording for {duration} seconds (or press Ctrl+C to stop early)...")
    
    # Store audio data chunks collected by the callback function
    audio_data_chunks = []

    def callback(indata, frames, time, status):
        """This is called (from a separate thread) for each audio block."""
        if status:
            print(f"Audio stream warning: {status}")
        # Append the incoming audio data to the list
        audio_data_chunks.append(indata.copy())

    # Calculate total frames for the duration
    num_samples = int(duration * SAMPLERATE)
    
    # Start the input stream
    with sd.InputStream(samplerate=SAMPLERATE, channels=1, dtype='int16', callback=callback) as stream:
        
        try:
            # Block the main thread, allowing the input stream to run for the full duration
            sd.sleep(duration * 1000)
            
        except KeyboardInterrupt:
            # User pressed Ctrl+C
            print("\nRecording manually stopped (Ctrl+C).")
        
        except Exception as e:
            print(f"\nAn audio error occurred: {e}")
            return None
        
        # Stream automatically stops when exiting the 'with' block
        
    # Concatenate all recorded chunks into a single numpy array
    if audio_data_chunks:
        recorded_audio = np.concatenate(audio_data_chunks, axis=0)
        print(f"✅ Captured {len(recorded_audio) / SAMPLERATE:.2f} seconds of audio.")
        return recorded_audio
    else:
        print("No audio data captured.")
        return None


def save_and_transcribe(audio, speaker):
    """Saves audio chunk, transcribes it, and prints the result."""
    
    if os.path.exists(TEMP_FILENAME):
        os.remove(TEMP_FILENAME)
        
    # Save the captured audio data to a WAV file
    write(TEMP_FILENAME, SAMPLERATE, audio)

    print("🌀 Transcribing...")
    
    try:
        # Run transcription using faster-whisper
        segments, info = model.transcribe(TEMP_FILENAME, beam_size=5)

        # Print the results to the conversation log format
        print(f"\n🗣️ {speaker}:")
        full_text = ""
        for segment in segments:
            full_text += segment.text.strip() + " "
        
        print(f"    {full_text.strip()}")
        
    except Exception as e:
        print(f"Transcription Error: {e}")
    finally:
        # Clean up the temporary file
        if os.path.exists(TEMP_FILENAME):
            os.remove(TEMP_FILENAME)

# --- Main Program Loop ---
if __name__ == "__main__":
    print("\n" + "="*50)
    print("      Sales Dialogue Transcriber (Python)")
    print("="*50)
    print(f"Max Recording Time per turn: {MAX_DURATION} seconds")
    print("Press Ctrl+C to exit the script.")

    # Conversation loop
    while True:
        try:
            # 1. Speaker Selection
            speaker_input = input("\nSelect Speaker (1=Customer, 2=Salesperson, q=Quit): ").strip().lower()
            
            if speaker_input == 'q':
                break
            
            if speaker_input == '1':
                current_speaker = "Customer"
                print("Starting Customer turn...")
            elif speaker_input == '2':
                current_speaker = "Salesperson"
                print("Starting Salesperson turn...")
            else:
                print("Invalid selection. Please choose 1, 2, or q.")
                continue

            # 2. Recording
            recorded_audio = record_audio(MAX_DURATION)
            
            # 3. Transcription
            if recorded_audio is not None:
                save_and_transcribe(recorded_audio, current_speaker)
            
            time.sleep(1) # Small pause before next turn

        except KeyboardInterrupt:
            # Final exit if Ctrl+C is pressed outside of the recording function
            print("\n\n🛑 Dialogue transcription session ended.")
            break
        except Exception as e:
            print(f"\nAn unexpected error occurred in the main loop: {e}")
            break