# Real-time Cow Detection and Classification

This project is a web-based application that uses your device's camera to detect cows in real-time. When a cow is detected, it is highlighted on the video feed, and a snapshot is taken. This snapshot is then used for image classification to predict the breed of the cow.

## ✨ Features

- **Real-time Object Detection:** Detects cows in the camera feed using the COCO-SSD model.
- **Image Classification:** Classifies the detected cow using MobileNet and ResNet models.
- **Grad-CAM Visualization:** Generates a heatmap to visualize the parts of the image that were important for the classification.
- **Cow Counter:** Counts the number of cows detected.
- **Gallery:** Displays the captured images of the cows along with the classification predictions.

## 🛠️ Technologies Used

- **HTML5:** For the basic structure of the web page.
- **CSS:** For styling the user interface.
- **JavaScript:** For the application logic.
- **TensorFlow.js:** For running the machine learning models in the browser.
    - **COCO-SSD (lite_mobilenet_v2):** For object detection.
    - **MobileNet (v1, alpha: 0.25):** For image classification.

## 🚀 How to Run the Project

To run this project, you need a web server to serve the `index.html` and `app.js` files. You can use any simple web server for this purpose.

### Prerequisites

- A modern web browser that supports WebRTC (for camera access).
- An internet connection (to load the TensorFlow.js models).

### Steps to Run

1. **Clone or download the repository.**
2. **Navigate to the project directory.**
3. **Start a simple web server.** A simple way to do this is to use Python's built-in HTTP server.

   - **For Python 3:**
     ```bash
     python -m http.server
     ```

   - **For Python 2:**
     ```bash
     python -m SimpleHTTPServer
     ```

4. **Open your web browser and navigate to the address provided by the web server** (usually `http://localhost:8000`).

5. **Grant camera permission** when prompted by the browser.

## 📝 Notes

- The models used in this project have been updated to lighter versions for faster loading times. The original models were COCO-SSD and ResNet, which have been replaced with COCO-SSD with `lite_mobilenet_v2` and MobileNet v1.
- The classification of cow breeds is for demonstration purposes only. The model is not trained to distinguish between the specified cow breeds. It will show a deterministic breed from the provided list based on the image.
- The Grad-CAM heatmap is generated for the predicted class.

## 📁 File Descriptions

- **`index.html`:** The main HTML file that defines the structure of the web page.
- **`app.js`:** The JavaScript file that contains the application logic, including loading the models, accessing the camera, and performing the detection and classification.
- **`package-lock.json`:** This file is present but not strictly necessary for running the project, as the dependencies are loaded from a CDN.
