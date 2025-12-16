Deep Fake Radar
Seeing Beyond the Fake with Artificial Intelligence
Deep Fake Radar is an AI-based web application designed to detect whether a given video is REAL or FAKE.
The system combines deep learning (CNN–LSTM) with temporal frame-consistency analysis to provide stable and reliable deepfake detection results.
📌 Project Overview
With the rapid growth of AI-generated and manipulated videos, it has become difficult to verify the authenticity of digital media.
Deep Fake Radar addresses this challenge by analyzing both spatial features (visual artifacts) and temporal inconsistencies across video frames.
This project is developed as a Final Year Engineering Project and demonstrates the end-to-end pipeline of:
Video upload
Frame extraction
AI-based inference
Result visualization through a web interface
✨ Features
📹 Upload and preview video files
🧠 CNN + LSTM based deepfake detection model
🔍 Temporal frame-consistency heuristic for stability
📊 Confidence score for predictions
🌐 Flask-based web application
🎨 Modern and responsive user interface
🛠️ Technologies Used
Backend & AI
Python
TensorFlow / Keras
OpenCV
NumPy
Scikit-learn
Frontend
HTML
CSS
JavaScript
Web Framework
Flask
📂 Project Structure
Deep-Fake-Radar/
│
├── app.py                     # Flask backend
├── train_model.py              # Model training script
├── requirements.txt            # Project dependencies
│
├── model/
│   └── deepfake_model.h5       # Trained CNN-LSTM model
│
├── templates/
│   └── index.html              # Frontend UI
│
├── uploads/                    # Uploaded videos (ignored in Git)
├── venv/                       # Virtual environment (ignored)
🚀 How to Run the Project
1️⃣ Clone the Repository
git clone https://github.com/your-username/Deep-Fake-Radar.git
cd Deep-Fake-Radar
2️⃣ Create Virtual Environment
python -m venv venv
venv\Scripts\activate    # Windows
3️⃣ Install Dependencies
pip install -r requirements.txt
4️⃣ Run the Application
python app.py
5️⃣ Open in Browser
http://127.0.0.1:5000
🧪 Model Description
CNN (Convolutional Neural Network)
Extracts spatial facial and visual features from each frame.
LSTM (Long Short-Term Memory)
Captures temporal dependencies and inconsistencies across video frames.
Hybrid Fusion Approach
Combines deep learning prediction with frame-consistency analysis to improve robustness and reduce false predictions.
Note: For demonstration purposes, the model is trained using simulated data.
The architecture is designed to support large-scale datasets such as FaceForensics++, DFDC, and Celeb-DF in future work.
🎓 Academic Note
This project is developed strictly for educational and research purposes as part of a final-year academic submission.
It demonstrates system design, AI integration, and deployment rather than production-level accuracy.
🔮 Future Enhancements
Training on real-world deepfake datasets
Face detection and alignment before analysis
Audio-based deepfake detection
Cloud-based deployment with GPU support
Detailed analytics and frame-wise visualization
👤 Author
Shreyansh Pathak, Sudhanshu Mandloi , Om Vishwakarma
Final Year Engineering Students
Project: Deep Fake Radar
📜 License
This project is intended for academic use only.
