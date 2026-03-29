# NeuralNumber — Handwritten Digit Identification

A professional handwritten digit recognition system built from scratch using **NumPy** and **Python**.

## 🚀 Performance
- **98.71% Accuracy** on MNIST.
- **Adam Optimizer** for improved convergence.
- **Data Augmentation** (random pixel shifts) for real-world robustness.

## 🔬 Features
- **Neural Network from Scratch**: Implemented `LinearLayer`, `Relu`, `Softmax`, `CrossEntropyLoss`, and `Adam` without high-level ML libraries.
- **FastAPI Backend**: Real-time identification of digits from the browser canvas.
- **1970s Instrument UI**: A custom, vintage-inspired web interface with CRT effects and phosphor displays.
- **Light/Dark Mode**: Persistent "Day" (Olive/Cream) and "Night" (Amber/Charcoal) themes.

## 🛠️ Tech Stack
- **Math**: NumPy
- **Image Processing**: PIL (Pillow), Scipy
- **Backend**: FastAPI, Uvicorn
- **Frontend**: Vanilla HTML/CSS/JS (Vintage Instrument Design)

## 🏃 How to Run
1.  **Install dependencies**:
    ```bash
    pip install numpy scipy fastapi uvicorn pillow scikit-learn
    ```
2.  **Prepare Data**:
    ```bash
    python data_loader.py
    ```
3.  **Train Model**:
    ```bash
    python train.py
    ```
4.  **Launch Web App**:
    ```bash
    python -m uvicorn app:app --reload
    ```
5.  **Access**: Open `http://localhost:8000` in your browser.

## 📂 Project Structure
- `neural_network.py`: The core engine (Layers, Optimizer, Container).
- `train.py`: Training loop with augmentation and Adam.
- `app.py`: FastAPI server for real-time predictions.
- `static/`: Frontend assets (1970s instrument UI).
- `data_loader.py`: MNIST downloader and processor.
