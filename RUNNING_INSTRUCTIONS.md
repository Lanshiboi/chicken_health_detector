# Running the Chicken Health Detector Project

## Prerequisites
- Python 3.x installed
- All required Python packages installed (Flask, TensorFlow, OpenCV, NumPy)
  - You can install them using:
    ```
    pip install -r requirements.txt
    ```

## Steps to Run

### 1. Start the Backend Server
- Open a terminal in the project directory.
- Run the Flask backend server by executing:
  ```
  python backend.py
  ```
- The server will start running at `http://127.0.0.1:5000`.

### 2. Open the Frontend
- Open the `analyze.html` file in your web browser.
  - You can open it directly by double-clicking the file.
  - Or serve it via a local HTTP server for better compatibility:
    - Using Python 3:
      ```
      python -m http.server 8000
      ```
    - Then open `http://localhost:8000/analyze.html` in your browser.

### 3. Use the Application
- On the Analyze page, upload a chicken thermal image.
- Click the Analyze button to send the image to the backend for analysis.
- View the analysis results displayed on the page.
- Optionally, save the analysis.

## Notes
- Ensure the backend server is running before analyzing images.
- The backend uses the trained model to predict chicken health status.

If you need help with any step, feel free to ask.
