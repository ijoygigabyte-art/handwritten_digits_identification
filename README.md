# NeuralNumber

A professional handwritten digit recognition system built from scratch using **NumPy**.

## 🚀 Performance
- **98.71% Accuracy** on MNIST.
- **Adam Optimizer** & **Data Augmentation** for robustness.

## 🏃 How to Run

### Local (Manual)
1.  **Install**: `pip install -r requirements.txt`
2.  **Prepare**: `python data_loader.py`
3.  **Train**: `python train.py`
4.  **Launch**: `python -m uvicorn app:app --reload`
5.  **Access**: `http://localhost:8000`

### Docker (Recommended)
1.  **Build & Run**: `docker-compose up --build`
2.  **Access**: `http://localhost:8000`

## 🛠️ Features
- **1970s Instrument UI**: Phosphor green CRT display and analog meter bars.
- **Light/Dark Mode**: Persistent "Day" (Olive) and "Night" (Amber) themes.
- **Custom NN Engine**: All layers and optimizers built without ML frameworks.

## 💻 Tech Stack
- **Core Engine**: NumPy, SciPy (Linear Algebra, Optimization from scratch)
- **Backend API**: FastAPI, Uvicorn
- **Frontend UI**: Vanilla JS, HTML5 Canvas, Advanced CSS3 (for CRT effects)
- **DevOps**: Docker, Docker Compose (Reproducible environments)
- **Host Platform**: Render (Automatic CD/CI)

Access: https://handwritten-digits-identification.onrender.com
