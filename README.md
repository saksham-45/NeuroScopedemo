# NeuroScope: Real-Time Emotion Detection

![NeuroScope Screenshot](https://placehold.co/800x400/D1D5DB/4B5563?text=NeuroScope+App+Screenshot+Placeholder)
*(Replace the placeholder above with an actual screenshot of your application)*

NeuroScope is a desktop application built with Python and PyQt5 that performs real-time emotion detection using a computer's webcam feed. It leverages MTCNN for accurate face detection and a pre-trained TensorFlow/Keras model to classify facial expressions into one of seven emotions: Angry, Disgust, Fear, Happy, Sad, Surprise, or Neutral. Detected emotions and their confidence scores are logged to a local SQLite database and visualized in a real-time graph.

## Features

* **Real-Time Video Feed:** Displays the live feed from the default webcam.
* **Face Detection:** Uses the Multi-task Cascaded Convolutional Networks (MTCNN) algorithm to detect faces in the video stream.
* **Emotion Recognition:** Classifies the emotion of detected faces using a convolutional neural network (CNN) model.
* **On-Screen Display:** Overlays bounding boxes around detected faces and displays the predicted emotion label along with its confidence score.
* **Data Logging:** Records detected emotions, confidence scores, and timestamps into an SQLite database (`emotion_data.db`).
* **Real-Time Graphing:** Visualizes the confidence scores of detected emotions over the last ~50 detections using Matplotlib.
* **Cross-Platform:** Built with standard libraries, aiming for compatibility (tested primarily with considerations for macOS).
* **GPU Acceleration:** Includes configuration attempts for TensorFlow GPU acceleration (specifically Metal for macOS).

## Requirements

* Python 3.7+
* Webcam
* Pre-trained emotion detection model file (`emotion_detection_model.h5`)

## Dependencies

The application relies on the following Python libraries:

* `PyQt5`: For the graphical user interface.
* `opencv-python`: For video capture and image processing.
* `tensorflow`: For loading and running the emotion detection model.
* `mtcnn`: For the face detection algorithm.
* `matplotlib`: For plotting the emotion confidence graph.
* `numpy`: For numerical operations.

For macOS users seeking GPU acceleration:
* `tensorflow-metal`: (Optional, for Apple Silicon GPUs)

## Installation

1.  **Clone the repository (or download the source code):**
    ```bash
    git clone <your-repository-url>
    cd neuroscope # Or your project directory name
    ```

2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3.  **Install the required dependencies:**
    ```bash
    pip install PyQt5 opencv-python tensorflow mtcnn matplotlib numpy
    ```
    * **Note for macOS GPU:** If you have an Apple Silicon Mac and want GPU acceleration, follow TensorFlow's official instructions to install `tensorflow-metal`: [https://developer.apple.com/metal/tensorflow-plugin/](https://developer.apple.com/metal/tensorflow-plugin/)

4.  **Place the Model File:** Ensure the pre-trained emotion detection model file, named `emotion_detection_model.h5`, is placed in the same directory as the `neuroscope_fixed.py` script. You might need to download this model separately if it's not included in the repository.

## Usage

1.  Make sure your webcam is connected and accessible.
2.  Navigate to the project directory in your terminal.
3.  Activate the virtual environment (if you created one).
4.  Run the application script:
    ```bash
    python neuroscope_fixed.py
    ```
5.  The application window should appear, displaying the webcam feed with detected faces, emotions, and the real-time graph.
6.  Close the application window to stop the process. Emotion data will be saved in `emotion_data.db`.

## Configuration

The following constants can be adjusted within the `neuroscope_fixed.py` script:

* `MODEL_PATH`: Path to the emotion detection model file (default: `'emotion_detection_model.h5'`).
* `EMOTION_LABELS`: List of emotion labels corresponding to the model's output (default: `['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']`).
* `MODEL_INPUT_SIZE`: Expected input image dimensions for the model (default: `(48, 48)`).
* `DATABASE_FILE`: Name of the SQLite database file (default: `'emotion_data.db'`).
* `FRAME_SKIP`: Process every Nth frame to potentially improve performance (default: `2`). Set to `1` to process every frame.
* `GRAPH_UPDATE_INTERVAL`: How often the graph updates in milliseconds (default: `500`).
* `MAX_GRAPH_POINTS`: Maximum number of data points shown on the graph (default: `50`).

## Database

The application creates and uses an SQLite database file (`emotion_data.db` by default) in the same directory. The `emotions` table stores the following information:

* `timestamp`: REAL (Unix timestamp of the detection)
* `emotion`: TEXT (Detected emotion label)
* `confidence`: REAL (Confidence score of the detection, 0.0 to 1.0)

You can use any standard SQLite browser to view the logged data.

## Troubleshooting

* **No GPU Detected (macOS):** Ensure `tensorflow-metal` is correctly installed. Verify your TensorFlow installation supports Metal. Check console output for specific errors during TensorFlow initialization.
* **Camera Not Opening:** Verify the correct camera index is used in `cv2.VideoCapture(0)`. Ensure no other application is using the camera. Check camera permissions for the application/terminal.
* **Model Not Found:** Make sure `emotion_detection_model.h5` exists in the correct directory and the `MODEL_PATH` variable is set correctly.
* **Low Performance:** Increase `FRAME_SKIP` to process fewer frames. Ensure you are using GPU acceleration if available. Close other resource-intensive applications.
* **MTCNN Errors:** MTCNN can sometimes be sensitive to library versions (TensorFlow, Keras, etc.). Ensure compatible versions are installed. Check the MTCNN documentation or GitHub issues for known problems.

## License

*(Specify your license here, e.g., MIT License, Apache 2.0, etc. If unsure, you can state "License TBD" or consult choosealicense.com)*

Example:
This project is licensed under the MIT License - see the LICENSE.md file for details.
## Acknowledgements

* Mention any libraries, datasets, or research papers that significantly contributed to the project.
* Hat tip to the creators of PyQt, OpenCV, TensorFlow, MTCNN, and Matplotlib.
