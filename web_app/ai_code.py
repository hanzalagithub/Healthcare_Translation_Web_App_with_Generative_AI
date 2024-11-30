import requests
from pydub import AudioSegment
from translate import Translator
from gtts import gTTS
import os
import tempfile

# Hugging Face API Details
API_URL = "https://api-inference.huggingface.co/models/openai/whisper-large-v3-turbo"
HEADERS = {"Authorization": "Bearer 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"}

# Function to compress the audio file
def compress_audio(input_audio_path, output_audio_path, target_sample_rate=16000):
    try:
        audio = AudioSegment.from_file(input_audio_path)
        # Convert to mono and downsample to target sample rate
        audio = audio.set_frame_rate(target_sample_rate).set_channels(1)
        # Export the compressed audio
        audio.export(output_audio_path, format="wav")
        print(f"Compressed audio saved to {output_audio_path}")
    except Exception as e:
        print(f"Error during audio compression: {e}")

# Function to query Hugging Face API
def query(audio_path):
    try:
        with open(audio_path, "rb") as f:
            data = f.read()
        response = requests.post(API_URL, headers=HEADERS, data=data)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        return {"error": str(e)}

# Function to split text into chunks of 500 characters or less
def split_text_into_chunks(text, max_length=500):
    # Split text into chunks of max_length characters
    return [text[i:i+max_length] for i in range(0, len(text), max_length)]

# Function to translate text
def translate_text_free(text, target_language):
    translator = Translator(to_lang=target_language)
    return translator.translate(text)

# Function to convert text to speech
def text_to_speech(text, language, output_path="output.mp3"):
    try:
        tts = gTTS(text=text, lang=language, slow=False)
        tts.save(output_path)
        print(f"Audio saved to {output_path}")
        return output_path
    except Exception as e:
        print(f"Error during text-to-speech conversion: {e}")
        return None

# Main script
if __name__ == "__main__":
    # Input and output file paths
    input_audio = "sample.wav"  # Replace with your input file
    compressed_audio = "compressed_sample.wav"

    print("Compressing audio...")
    compress_audio(input_audio, compressed_audio)

    print("Processing compressed audio with API...")
    output = query(compressed_audio)

    if "text" in output:
        transcribed_text = output["text"]  # Extract the transcription string
        print("Transcription:")
        print(transcribed_text)

        # Step 1: Split the transcribed text into chunks
        chunks = split_text_into_chunks(transcribed_text)

        # Step 2: Translate each chunk
        translated_chunks = [translate_text_free(chunk, "es") for chunk in chunks]

        # Step 3: Combine all translated chunks
        final_translated_text = " ".join(translated_chunks)
        print("Translated Text:")
        print(final_translated_text)

        # Step 4: Convert the translated text to speech
        audio_output = text_to_speech(final_translated_text, "es")

        # Step 5: Play the audio if successfully created
        if audio_output:
            print("Playing the audio...")
            os.system(f"mpg321 {audio_output}")  # Replace mpg321 with your audio player if needed
    else:
        print("Error:")
        print(output.get("error", "Unknown error"))

    # Clean up the compressed file if needed
    try:
        os.remove(compressed_audio)
    except Exception as e:
        print(f"Error deleting compressed file: {e}")