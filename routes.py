import os
import time
import traceback
import yt_dlp
import requests
from flask import Blueprint, request, jsonify, send_from_directory
from dotenv import load_dotenv
import azure.cognitiveservices.speech as speechsdk
from moviepy.editor import VideoFileClip
# Create Flask application instance
from app import app

# Load environment variables from .env file
load_dotenv()

routes = Blueprint('routes', __name__)
UPLOAD_FOLDER = r'backend\backend\uploads'

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# YouTube to WAV configuration
ydl_opts = {
    'format': 'bestaudio/best',
    'postprocessors': [{
        'key': 'FFmpegExtractAudio',
        'preferredcodec': 'wav',
    }],
}

# Function to download audio from YouTube URL and convert to WAV
def download_from_url(url):
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        result = ydl.extract_info(url, download=True)
        return ydl.prepare_filename(result).rsplit('.', 1)[0] + '.wav'

# Function to convert video file to WAV audio
def video_to_wav(video_file):
    try:
        output_wav = os.path.splitext(video_file)[0] + '.wav'
        video = VideoFileClip(video_file)
        audio = video.audio
        audio.write_audiofile(output_wav)
        audio.close()
        video.close()
        return output_wav
    except Exception as e:
        print(f"Error converting video to WAV: {str(e)}")
        return None

# Speech SDK Callbacks
def conversation_transcriber_recognition_canceled_cb(evt: speechsdk.SessionEventArgs):
    print('Canceled event')

def conversation_transcriber_session_stopped_cb(evt: speechsdk.SessionEventArgs):
    print('SessionStopped event')

def conversation_transcriber_transcribed_cb(evt: speechsdk.SpeechRecognitionEventArgs, transcribed_text):
    try:
        if evt.result.reason == speechsdk.ResultReason.RecognizedSpeech:
            transcribed_text.append(f"Speaker {evt.result.speaker_id}: {evt.result.text}\n")
    except Exception as e:
        traceback.print_exc()
        print("Error in transcription callback:", e)

def conversation_transcriber_session_started_cb(evt: speechsdk.SessionEventArgs):
    print('SessionStarted event')

# Function to perform speech recognition
def recognize_from_file(audio_file):
    speech_config = speechsdk.SpeechConfig(subscription=os.environ.get('SPEECH_KEY'), region=os.environ.get('SPEECH_REGION'))
    audio_config = speechsdk.audio.AudioConfig(filename=audio_file)
    auto_detect_source_language_config = speechsdk.languageconfig.AutoDetectSourceLanguageConfig(languages=["hi-IN", "mr-IN", "te-IN", "gu-IN"])
    speech_recognizer = speechsdk.SpeechRecognizer(speech_config=speech_config, auto_detect_source_language_config=auto_detect_source_language_config, audio_config=audio_config)

    result = speech_recognizer.recognize_once()
    auto_detect_source_language_result = speechsdk.AutoDetectSourceLanguageResult(result)
    detected_language = auto_detect_source_language_result.language
    print("Detected language:", detected_language)

    # Check if the language was detected successfully
    if not detected_language:
        print("Language detection failed. Using default language: en-US")
        detected_language = "en-US"  # Fallback to English

    speech_config.speech_recognition_language = detected_language
    conversation_transcriber = speechsdk.transcription.ConversationTranscriber(speech_config=speech_config, audio_config=audio_config)
    transcribed_text = []

    transcribing_stop = False

    def stop_cb(evt: speechsdk.SessionEventArgs):
        print('CLOSING on {}'.format(evt))
        nonlocal transcribing_stop
        transcribing_stop = True

    conversation_transcriber.transcribed.connect(lambda evt: conversation_transcriber_transcribed_cb(evt, transcribed_text))
    conversation_transcriber.session_started.connect(conversation_transcriber_session_started_cb)
    conversation_transcriber.session_stopped.connect(conversation_transcriber_session_stopped_cb)
    conversation_transcriber.canceled.connect(conversation_transcriber_recognition_canceled_cb)
    conversation_transcriber.session_stopped.connect(stop_cb)
    conversation_transcriber.canceled.connect(stop_cb)

    conversation_transcriber.start_transcribing_async()

    while not transcribing_stop:
        time.sleep(.5)

    conversation_transcriber.stop_transcribing_async()
    print("Transcription completed successfully.")

    return ''.join(transcribed_text)

# Function to perform translation using OpenAI
def translate_text(text, chunk_size=100):
    words = text.split()
    translated_chunks = []
    
    # Break the text into chunks of the specified size
    for i in range(0, len(words), chunk_size):
        chunk = ' '.join(words[i:i + chunk_size])
        print(f"Translating chunk {i // chunk_size + 1}/{len(words) // chunk_size + 1}: {chunk[:50]}...")
        prompt = "Translate the following text to English:"
        output = get_openai_response(prompt, chunk)
        if output:
            print(f"Translation for chunk {i // chunk_size + 1}/{len(words) // chunk_size + 1} completed: {output[:50]}...")
            translated_chunks.append(output)
        else:
            print(f"Failed to get translation for chunk {i // chunk_size + 1}/{len(words) // chunk_size + 1}.")
    
    return ' '.join(translated_chunks)

# Function to request completion from OpenAI API
def get_openai_response(prompt, input_text):
    api_key = os.getenv('AZURE_OPENAI_API_KEY')
    api_base_url = os.getenv('AZURE_OPENAI_ENDPOINT')
    
    api_version = '2022-12-01'
    deployment_id = 'ver01'
    api_url = f"{api_base_url}/openai/deployments/{deployment_id}/completions?api-version={api_version}"
    
    payload = {
        "prompt": f"{prompt}\n\nInput: {input_text}\n\nOutput:",
        "max_tokens": 2000,
        "temperature": 0.7,
        "top_p": 0.95,
        "n": 1,
        "stop": ["\n"]
    }
    headers = {
        "Content-Type": "application/json",
        "api-key": api_key
    }

    try:
        print(f"Request Payload: {payload}")
        response = requests.post(api_url, json=payload, headers=headers)
        response.raise_for_status()
        completion = response.json()
        if "choices" in completion and len(completion["choices"]) > 0:
            return completion["choices"][0]["text"].strip()
        else:
            print("Unexpected response format or empty response:", completion)
            return None
    except requests.RequestException as e:
        print("Request failed:", e)
        if e.response is not None:
            print("Response status code:", e.response.status_code)
            print("Response content:", e.response.text)
        return None

@routes.route('/process_audio', methods=['POST'])
def process_audio():
    choice = request.form['choice']
    
    if choice == '1':
        if 'url' in request.form:
            url = request.form['url']
            audio_file = download_from_url(url)
        else:
            return jsonify({'error': 'URL is missing.'}), 400
    elif choice == '2':
        if 'audio_file' in request.files:
            audio_file = request.files['audio_file']
            audio_file.save(os.path.join(UPLOAD_FOLDER, audio_file.filename))
            audio_file = os.path.join(UPLOAD_FOLDER, audio_file.filename)
        else:
            return jsonify({'error': 'Audio file is missing.'}), 400
    elif choice == '3':
        if 'video_file' in request.files:
            video_file = request.files['video_file']
            video_file.save(os.path.join(UPLOAD_FOLDER, video_file.filename))
            video_file_path = os.path.join(UPLOAD_FOLDER, video_file.filename)
            audio_file = video_to_wav(video_file_path)
        else:
            return jsonify({'error': 'Video file is missing.'}), 400
    else:
        return jsonify({'error': 'Invalid choice.'}), 400

    transcribed_text = recognize_from_file(audio_file)
    
    if transcribed_text:
        translated_text = translate_text(transcribed_text)
        
        if translated_text:
            # Save output to text files with specific names
            transcribed_file = os.path.join(UPLOAD_FOLDER, 'transcribed.txt')
            translated_file = os.path.join(UPLOAD_FOLDER, 'translated.txt')
            with open(transcribed_file, 'w', encoding='utf-8') as f:
                f.write(transcribed_text)
            with open(translated_file, 'w', encoding='utf-8') as f:
                f.write(translated_text)
            
            # Return JSON response with transcribed and translated texts
            return jsonify({
                'transcribed_text': transcribed_text,
                'translated_text': translated_text,
                'transcribed_file': 'transcribed.txt',
                'translated_file': 'translated.txt'
            })
        else:
            return jsonify({'error': 'Translation failed.'}), 500
    else:
        return jsonify({'error': 'Transcription failed.'}), 500

@routes.route('/download/<filename>')
def download_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)

@routes.route('/ask', methods=['POST'])
def ask():
    try:
        # Get the question from the form data
        data = request.get_json()
        question = data.get('question')
        
        if not question:
            return jsonify({'error': 'Question is missing.'}), 400

        # Define the path to the translated.txt file
        file_path = os.path.join(UPLOAD_FOLDER, 'translated.txt')

        # Read the document text from the translated.txt file
        document_text = read_document(file_path)
        if not document_text:
            return jsonify({'error': 'Failed to read document text.'}), 500

        # Create the prompt with the question and document text
        prompt = "Just answer the following question very accurately:"
        input_text = f"{question}\n\n{document_text}"
        answer = get_openai_response(prompt, input_text)

        if answer:
            return jsonify({'answer': answer})
        else:
            return jsonify({'error': 'Failed to get response from OpenAI API.'}), 500
    except Exception as e:
        print(f"Error in /ask route: {str(e)}")
        return jsonify({'error': 'Internal server error.'}), 500

# Helper function to read document text (if needed)
def read_document(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()

@app.route('/read_texts', methods=['GET'])
def read_texts():
    try:
        transcribed_file_path = os.path.join(UPLOAD_FOLDER, 'transcribed.txt')
        translated_file_path = os.path.join(UPLOAD_FOLDER, 'translated.txt')

        transcribed_text = read_document(transcribed_file_path)
        translated_text = read_document(translated_file_path)

        # Print transcribed_text to terminal
        print("Transcribed Text:")
        print(transcribed_text)

        return jsonify({
            'transcribed_text': transcribed_text,
            'translated_text': translated_text
        })
    except Exception as e:
        print(f"Error in /read_texts route: {str(e)}")
        return jsonify({'error': 'Failed to read texts.'}), 500
