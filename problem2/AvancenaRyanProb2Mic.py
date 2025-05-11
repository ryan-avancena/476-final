import whisper
import sounddevice as sd
import numpy as np
import json
from datetime import datetime

model = whisper.load_model("base")
samplerate = 16000  # Whisper expects 16kHz
block_duration = 5  # seconds per block

all_transcriptions = []

def callback(indata, frames, time, status):
    if status:
        print("Status:", status)

    audio_np = indata[:, 0]  # mono
    audio_float32 = audio_np.astype(np.float32)

    if np.max(np.abs(audio_float32)) > 1:
        audio_float32 = audio_float32 / np.max(np.abs(audio_float32))

    print("Transcribing...")
    result = model.transcribe(audio_float32, language="en")

    timestamp = datetime.now().isoformat()
    text = result["text"]
    segments = result.get("segments", [])  # includes start and end time

    block_data = {
        "timestamp": timestamp,
        "text": text,
        "segments": [
            {
                "start": seg["start"],
                "end": seg["end"],
                "text": seg["text"]
            } for seg in segments
        ]
    }

    all_transcriptions.append(block_data)
    print("You said:", text)

print("Listening in blocks of", block_duration, "seconds. Press Ctrl+C to stop.")

try:
    with sd.InputStream(channels=1, samplerate=samplerate, dtype='float32',
                        blocksize=int(samplerate * block_duration),
                        callback=callback):
        while True:
            sd.sleep(int(block_duration * 1000))  # Keep the main thread alive

except KeyboardInterrupt:
    print("Stopped.")

    # Save to a JSON file when stopped
    with open("transcription_log.json", "w") as f:
        json.dump(all_transcriptions, f, indent=4)

    print("Transcriptions saved to 'transcription_log.json'.")