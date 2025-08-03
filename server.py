# server.py
import os
import subprocess
import uuid
from flask import Flask, request, send_from_directory, jsonify

# --- Configuration ---
# IMPORTANT: Update this to the absolute path of your main.py script
PATH_TO_MAIN_SCRIPT = "C:/Users/hashe/ai_equestrian/src/main.py" 
UPLOADS_FOLDER = "uploads"
RESULTS_FOLDER = "results"

app = Flask(__name__)
app.config['UPLOADS_FOLDER'] = UPLOADS_FOLDER
app.config['RESULTS_FOLDER'] = RESULTS_FOLDER

# Create directories if they don't exist
os.makedirs(UPLOADS_FOLDER, exist_ok=True)
os.makedirs(RESULTS_FOLDER, exist_ok=True)

@app.route('/upload', methods=['POST'])
def upload_and_process_video():
    """
    Handles video upload, processing, and returning the result.
    """
    if 'video' not in request.files:
        return jsonify({"error": "No video file provided"}), 400

    video_file = request.files['video']
    if video_file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    # 1. Save the incoming video to the 'uploads' folder with a unique name
    unique_id = str(uuid.uuid4())
    input_filename = f"{unique_id}.mp4"
    input_filepath = os.path.join(app.config['UPLOADS_FOLDER'], input_filename)
    video_file.save(input_filepath)

    print(f"Video saved to: {input_filepath}")
    
    # --- IMPORTANT MODIFICATION ---
    # We need to know where your script will save the result.
    # Let's assume it saves the processed video in the 'results' folder
    # with "_processed" appended to the original unique ID.
    output_filename = f"{unique_id}_processed.mp4"
    output_filepath = os.path.join(app.config['RESULTS_FOLDER'], output_filename)

    try:
        # 2. Run your main.py script using subprocess
        # This is just like running it from the command line.
        # We pass the input and expected output paths to your script.
        print(f"Running processing script on {input_filepath}...")
        
        # Command: python C:/path/to/main.py <input_video_path> <output_video_path>
        subprocess.run(
            ["python", PATH_TO_MAIN_SCRIPT, input_filepath, output_filepath],
            check=True, # This will raise an error if your script fails
            capture_output=True, # Captures stdout/stderr
            text=True
        )
        
        print(f"Processing complete. Result saved to: {output_filepath}")

        # 3. Send the processed file back to the mobile app
        return send_from_directory(app.config['RESULTS_FOLDER'], output_filename, as_attachment=True)

    except subprocess.CalledProcessError as e:
        # If your script has an error, this will catch it.
        print(f"Error during processing: {e}")
        print(f"Script Stderr: {e.stderr}")
        return jsonify({"error": "Failed to process video.", "details": e.stderr}), 500
    except FileNotFoundError:
        return jsonify({"error": f"Result file not found at {output_filepath}. Did the script save it correctly?"}), 500
    finally:
        # 4. (Optional but recommended) Clean up the temporary files
        if os.path.exists(input_filepath):
            os.remove(input_filepath)
        # You might want to delay deleting the result file or handle it differently
        # if os.path.exists(output_filepath):
        #     os.remove(output_filepath)


if __name__ == '__main__':
    # Runs the server. '0.0.0.0' makes it accessible on your local network.
    app.run(host='0.0.0.0', port=5000, debug=True)