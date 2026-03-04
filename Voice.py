import asyncio # for asynchronous programming
import edge_tts # For text to speech conversion in Microsoft Edge 
import sounddevice as sd # For playing audio
import numpy as np # For handling audio data
import io # For handling audio data in memory
import soundfile as sf # For reading and writing audio files


async def speak(text, voice="en-AU-NatashaNeural"): # Selecting a voice for speech output
    communicate = edge_tts.Communicate(text, voice)
    
    audio_bytes = b""
    async for chunk in communicate.stream():
        if chunk["type"]=="audio":
            audio_bytes+=chunk["data"]
    with sf.SoundFile(io.BytesIO(audio_bytes)) as f:
        data = f.read(dtype="float32")
        sd.play(data, f.samplerate)
        sd.wait()
        