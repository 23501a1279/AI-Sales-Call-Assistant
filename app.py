import os
import pandas as pd
from faster_whisper import WhisperModel

#  Folder containing your audio files
audio_folder = "/Volumes/C/Task1-Infosys"

#  Load Whisper model
print("🔁 Loading Whisper model...")
model = WhisperModel("tiny", device="cpu")

#  Collect all .wav files in the folder
audio_files = sorted([f for f in os.listdir(audio_folder) if f.endswith(".wav")])

if not audio_files:
    print("⚠️ No .wav files found in folder!")
    exit()

#  Prepare a list to store all transcripts
transcripts = []

#  Process each audio file
for file_name in audio_files:
    audio_path = os.path.join(audio_folder, file_name)
    print(f"\n🔊 Processing: {file_name}")

    segments, info = model.transcribe(audio_path, beam_size=5)

    print("\n🗣️ Transcription:\n")

    # Alternate speakers
    speaker = "Salesperson"
    for segment in segments:
        start = round(segment.start, 2)
        end = round(segment.end, 2)
        text = segment.text.strip()

        print(f"{speaker} [{start}s - {end}s]: {text}")
        transcripts.append({
            "File": file_name,
            "Speaker": speaker,
            "Start Time": start,
            "End Time": end,
            "Text": text
        })

        # alternate speakers each segment
        speaker = "Customer" if speaker == "Salesperson" else "Salesperson"

#  Save all transcripts to CSV
output_csv = os.path.join(audio_folder, "Transcripts.csv")
pd.DataFrame(transcripts).to_csv(output_csv, index=False, encoding='utf-8')

print(f"\n✅ All done! Transcripts saved to: {output_csv}")