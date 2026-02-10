# # server.py
# import os
# import subprocess
# import uuid
# from flask import Flask, request, send_from_directory, jsonify

# # --- Configuration ---
# # IMPORTANT: Update this to the absolute path of your main.py script
# PATH_TO_MAIN_SCRIPT = os.path.join("src", "main.py")
# UPLOADS_FOLDER = "uploads"
# RESULTS_FOLDER = "results"

# app = Flask(__name__)
# app.config['UPLOADS_FOLDER'] = UPLOADS_FOLDER
# app.config['RESULTS_FOLDER'] = RESULTS_FOLDER

# # Create directories if they don't exist
# os.makedirs(UPLOADS_FOLDER, exist_ok=True)
# os.makedirs(RESULTS_FOLDER, exist_ok=True)

# @app.route('/upload', methods=['POST'])
# def upload_and_process_video():
#     """
#     Handles video upload, processing, and returning the result.
#     """
#     if 'video' not in request.files:
#         return jsonify({"error": "No video file provided"}), 400

#     video_file = request.files['video']
#     if video_file.filename == '':
#         return jsonify({"error": "No selected file"}), 400

#     # 1. Save the incoming video to the 'uploads' folder with a unique name
#     unique_id = str(uuid.uuid4())
#     input_filename = f"{unique_id}.mp4"
#     input_filepath = os.path.join(app.config['UPLOADS_FOLDER'], input_filename)
#     video_file.save(input_filepath)

#     print(f"Video saved to: {input_filepath}")
    
#     # --- IMPORTANT MODIFICATION ---
#     # We need to know where your script will save the result.
#     # Let's assume it saves the processed video in the 'results' folder
#     # with "_processed" appended to the original unique ID.
#     output_filename = f"{unique_id}_processed.mp4"
#     output_filepath = os.path.join(app.config['RESULTS_FOLDER'], output_filename)

#     try:
#         # 2. Run your main.py script using subprocess
#         # This is just like running it from the command line.
#         # We pass the input and expected output paths to your script.
#         print(f"Running processing script on {input_filepath}...")
        
#         # Command: python C:/path/to/main.py <input_video_path> <output_video_path>
#         subprocess.run(
#             ["python", PATH_TO_MAIN_SCRIPT, input_filepath, "0", "0"],  # '0' for debug and segment_flag
#             check=True, # This will raise an error if your script fails
#             capture_output=True, # Captures stdout/stderr
#             text=True
#         )
        
#         print(f"Processing complete. Result saved to: {output_filepath}")

#         # 3. Send the processed file back to the mobile app
#         return send_from_directory(app.config['RESULTS_FOLDER'], output_filename, as_attachment=True)

#     except subprocess.CalledProcessError as e:
#         # If your script has an error, this will catch it.
#         print(f"Error during processing: {e}")
#         print(f"Script Stderr: {e.stderr}")
#         return jsonify({"error": "Failed to process video.", "details": e.stderr}), 500
#     except FileNotFoundError:
#         return jsonify({"error": f"Result file not found at {output_filepath}. Did the script save it correctly?"}), 500
#     finally:
#         # 4. (Optional but recommended) Clean up the temporary files
#         if os.path.exists(input_filepath):
#             os.remove(input_filepath)
#         # You might want to delay deleting the result file or handle it differently
#         # if os.path.exists(output_filepath):
#         #     os.remove(output_filepath)


# if __name__ == '__main__':
#     # Runs the server. '0.0.0.0' makes it accessible on your local network.
#     app.run(host='0.0.0.0', port=5000, debug=True)

import os
import subprocess
import uuid
from flask import Flask, request, send_file, jsonify, Response
from flask_cors import CORS
from werkzeug.utils import secure_filename

# --- Configuration ---
PATH_TO_MAIN_SCRIPT = os.path.join("src", "main.py")
UPLOADS_FOLDER = "uploads"
RESULTS_FOLDER = "results"

app = Flask(__name__)
CORS(app)

app.config['UPLOADS_FOLDER'] = UPLOADS_FOLDER
app.config['RESULTS_FOLDER'] = RESULTS_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB max

# Create directories if they don't exist
os.makedirs(UPLOADS_FOLDER, exist_ok=True)
os.makedirs(RESULTS_FOLDER, exist_ok=True)


def send_video_file(filepath):
    """
    Send video file with proper streaming support for large files.
    """
    def generate():
        with open(filepath, 'rb') as f:
            while True:
                chunk = f.read(8192)  # 8KB chunks
                if not chunk:
                    break
                yield chunk
    
    file_size = os.path.getsize(filepath)
    
    return Response(
        generate(),
        mimetype='video/mp4',
        headers={
            'Content-Length': str(file_size),
            'Accept-Ranges': 'bytes',
            'Content-Type': 'video/mp4',
            'Cache-Control': 'no-cache'
        }
    )


@app.route('/upload', methods=['POST'])
def upload_and_process_video():
    """
    Handles video upload, processing, and returning the result.
    """
    print("\n" + "="*50)
    print("NEW UPLOAD REQUEST RECEIVED")
    print("="*50)
    
    if 'video' not in request.files:
        print("ERROR: No video file in request")
        return jsonify({"error": "No video file provided"}), 400

    video_file = request.files['video']
    if video_file.filename == '':
        print("ERROR: Empty filename")
        return jsonify({"error": "No selected file"}), 400

    # Generate unique ID
    unique_id = str(uuid.uuid4())
    input_filename = f"{unique_id}.mp4"
    input_filepath = os.path.join(app.config['UPLOADS_FOLDER'], input_filename)
    
    # Save uploaded file
    video_file.save(input_filepath)
    print(f"✓ Video saved to: {input_filepath}")
    print(f"  File size: {os.path.getsize(input_filepath) / (1024*1024):.2f} MB")
    
    # Define output path
    output_filename = f"{unique_id}_processed.mp4"
    output_filepath = os.path.join(app.config['RESULTS_FOLDER'], output_filename)

    try:
        # Run processing script
        print(f"▶ Running processing script...")
        
        result = subprocess.run(
            ["python", PATH_TO_MAIN_SCRIPT, input_filepath, "0", "0"],
            check=True,
            capture_output=True,
            text=True,
            timeout=300
        )
        
        print(f"✓ Processing complete!")
        if result.stdout:
            print(f"  Script output: {result.stdout[:200]}...")  # First 200 chars

        # Verify output file exists
        if not os.path.exists(output_filepath):
            print(f"✗ ERROR: Output file not found at {output_filepath}")
            return jsonify({
                "error": "Result file not found. Processing may have failed.",
                "details": "Output file was not created"
            }), 500

        # Check output file size
        output_size = os.path.getsize(output_filepath)
        print(f"✓ Output file created: {output_filepath}")
        print(f"  Output size: {output_size / (1024*1024):.2f} MB")
        
        if output_size == 0:
            print("✗ ERROR: Output file is empty (0 bytes)")
            return jsonify({
                "error": "Processed video is empty",
                "details": "Output file has 0 bytes"
            }), 500

        # Return success response
        video_url = f"https://api.aiequine.net/video/{output_filename}"
        print(f"✓ Returning video URL: {video_url}")
        print("="*50 + "\n")
        
        return jsonify({
            "processedVideoUrl": video_url,
            "analysis": {
                "overallScore": 85,
                "confidence": 92,
                "metrics": {
                    "headPosition": 88,
                    "shoulderAlignment": 82,
                    "backStraightness": 87,
                    "handPosition": 84,
                    "legPosition": 86,
                    "rhythmBalance": 83
                },
                "feedback": [
                    "Good overall posture maintained throughout the ride.",
                    "Work on keeping shoulders more level during transitions.",
                    "Excellent heel position and leg stability."
                ]
            }
        }), 200

    except subprocess.TimeoutExpired:
        print("✗ ERROR: Processing timeout (>5 minutes)")
        return jsonify({
            "error": "Processing took too long (timeout).",
            "details": "Video processing exceeded 5 minutes"
        }), 500
        
    except subprocess.CalledProcessError as e:
        print(f"✗ ERROR: Script failed with exit code {e.returncode}")
        print(f"  Stderr: {e.stderr}")
        return jsonify({
            "error": "Failed to process video.",
            "details": e.stderr if e.stderr else str(e)
        }), 500
        
    except Exception as e:
        print(f"✗ ERROR: Unexpected error: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({
            "error": "Unexpected error occurred.",
            "details": str(e)
        }), 500
        
    finally:
        # Clean up input file
        if os.path.exists(input_filepath):
            os.remove(input_filepath)
            print(f"✓ Cleaned up input file")


@app.route('/video/<filename>', methods=['GET'])
def serve_video(filename):
    """
    Serves video file for download.
    """
    filename = secure_filename(filename)
    filepath = os.path.join(app.config['RESULTS_FOLDER'], filename)
    
    if not os.path.exists(filepath):
        return jsonify({"error": "Video not found"}), 404
    
    print(f"📥 Serving video for download: {filename}")
    
    # Send file as attachment (triggers download)
    response = send_file(
        filepath,
        mimetype='video/mp4',
        as_attachment=True,  # This forces download instead of streaming
        download_name=f"SARA_Analysis_{filename}"
    )
    
    # CORS headers
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Cache-Control'] = 'no-cache'
    
    return response


@app.route('/health', methods=['GET'])
def health_check():
    """
    Health check endpoint.
    """
    uploads_files = len(os.listdir(UPLOADS_FOLDER)) if os.path.exists(UPLOADS_FOLDER) else 0
    results_files = len(os.listdir(RESULTS_FOLDER)) if os.path.exists(RESULTS_FOLDER) else 0
    
    return jsonify({
        "status": "healthy",
        "server": "waitress",
        "uploads_folder": UPLOADS_FOLDER,
        "results_folder": RESULTS_FOLDER,
        "uploads_exist": os.path.exists(UPLOADS_FOLDER),
        "results_exist": os.path.exists(RESULTS_FOLDER),
        "uploads_count": uploads_files,
        "results_count": results_files,
        "cwd": os.getcwd()
    }), 200


@app.route('/', methods=['GET'])
def root():
    """
    Root endpoint.
    """
    return jsonify({
        "message": "SARA API Server",
        "version": "1.0",
        "endpoints": {
            "upload": "/upload (POST)",
            "video": "/video/<filename> (GET)",
            "health": "/health (GET)"
        }
    }), 200


if __name__ == '__main__':
    print("\n" + "="*60)
    print("🐴 SARA API Server Starting...")
    print("="*60)
    print(f"📁 Working directory: {os.getcwd()}")
    print(f"📁 Uploads folder: {os.path.abspath(UPLOADS_FOLDER)}")
    print(f"📁 Results folder: {os.path.abspath(RESULTS_FOLDER)}")
    print(f"🌐 Server will run on: http://0.0.0.0:5000")
    print("="*60 + "\n")
    
    # Don't run with Flask dev server, use waitress
    from waitress import serve
    serve(app, host='0.0.0.0', port=5000, threads=4)