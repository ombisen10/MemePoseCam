# Gesture Meme

Computer vision project that detects facial and hand gestures in real time using the webcam and displays the corresponding meme.

## Technologies

- **Python 3.10.0**
- **OpenCV**
- **MediaPipe**
- **NumPy**

## Requirements

- **Python 3.8 - 3.11** (MediaPipe is not compatible with newer versions)

Install the dependencies with:

    pip install opencv-python mediapipe numpy

Or if you have the requirements file:

    pip install -r requirements.txt

## Detected gestures

| Gesture | Meme |
|---------|------|
| **Raised or furrowed eyebrows** | `perro.jpeg` |
| **Tongue out** | `gato1.png` |
| **Finger touching the mouth** | `cristiano.png` |
| **Two hands on the sides of the face** | `giphy.gif` |
| **Two hands above the nose** | `monkey.jpg` |
| **Index and middle finger extended** | `rata.jpeg` |

## Project structure

    gesture_meme/
    ├── main.py
    ├── requirements.txt
    ├── cara.jpeg
    ├── cristiano.png
    ├── gato1.png
    ├── giphy.gif
    ├── perro.jpeg
    ├── rata.jpeg
    └── monkey.jpg

## Usage

1. Clone the repository
2. Install the dependencies
3. Place the images in the same folder as `main.py`
4. Run:

       python main.py

5. When it starts, look straight ahead with a neutral face during **calibration**
6. Once calibrated, try the gestures in front of the camera
7. Press **ESC** to exit

## Notes

- **Calibration** takes a few seconds at startup and is necessary for the gestures to work correctly
- The images must be in the **same folder** as `main.py`
- Works best with **good lighting**
- Compatible with **Windows** (uses `CAP_DSHOW` for the camera)
