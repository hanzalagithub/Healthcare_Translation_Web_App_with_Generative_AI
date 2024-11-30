from flask import Flask, request, jsonify, render_template, send_from_directory
from werkzeug.utils import secure_filename
import os, shutil
import tempfile
from ai_code import compress_audio, query, split_text_into_chunks, translate_text_free, text_to_speech

app = Flask(__name__)

# Configure upload folder
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/process_audio", methods=["POST"])
def process_audio():
    if "audio_file" not in request.files:
        return jsonify({"error": "No audio file provided"}), 400
    
    audio_file = request.files["audio_file"]
    if audio_file.filename == "":
        return jsonify({"error": "No selected file"}), 400
    
    # Create a temporary directory to store the uploaded audio
    with tempfile.TemporaryDirectory() as tmp_dir:
        # Save the uploaded audio to the temporary file
        input_path = os.path.join(tmp_dir, "input_audio.mp3")
        audio_file.save(input_path)

        # Compress audio
        compressed_path = os.path.join(tmp_dir, "compressed_audio.mp3")
        compress_audio(input_path, compressed_path)

        # Query the Hugging Face API
        transcription_output = query(compressed_path)
        if "text" not in transcription_output:
            return jsonify({"error": transcription_output.get("error", "Failed to process audio")}), 500
        
        transcribed_text = transcription_output["text"]

        # Translate text
        target_language = request.form.get("target_language", "es")  # Default to Spanish
        chunks = split_text_into_chunks(transcribed_text)
        translated_chunks = [translate_text_free(chunk, target_language) for chunk in chunks]
        translated_text = " ".join(translated_chunks)

        # Convert translated text to speech
        output_audio_path = os.path.join(tmp_dir, "output.mp3")
        text_to_speech(translated_text, target_language, output_path=output_audio_path)

        # Save the output audio to the uploads folder
        output_audio_filename = os.path.basename(output_audio_path)
        output_audio_upload_path = os.path.join(app.config["UPLOAD_FOLDER"], output_audio_filename)
        # os.replace(output_audio_path, output_audio_upload_path)   #for local system outside the container
        shutil.move(output_audio_path, output_audio_upload_path)

        # Return the result as a JSON response
        return jsonify({
            "transcribed_text": transcribed_text,
            "translated_text": translated_text,
            "audio_output_url": f"/uploads/{output_audio_filename}"
        })

@app.route("/uploads/<filename>")
def serve_upload(filename):
    return send_from_directory(app.config["UPLOAD_FOLDER"], filename)

if __name__ == "__main__":
    app.run(debug=True)