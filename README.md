# 🌞 SolarDoc: AI-Powered Solar Panel Defect Detection

SolarDoc is an AI-powered system for **detecting and classifying defects in solar panels**.  
It leverages **deep learning** with a **ResNet50-based classifier** and **transfer learning** to identify common solar panel defects from images quickly and accurately.

---

## 🚀 Features

- Detects **5 common types** of solar panel defects:
  - 🧩 `broken`
  - 💡 `bright_spot`
  - ⚫ `black_border`
  - ✂️ `scratched`
  - ⚡ `non_electricity`
- Pre-trained **ResNet50** model with **transfer learning** for high accuracy
- Supports **checkpoint saving & resume training**
- Generates:
  - Training and validation **accuracy/loss plots**
  - **Confusion matrices**
  - **Prediction visualizations**
- Web interface built with **Flask + HTML**
- Ready-to-use **API endpoint** for uploading and classifying new images
- Can be connected with **Google Drive** for automatic model saving

---

## 🧠 Project Structure

📂 SolarDoc
┣ 📜 app.py # Flask web interface
┣ 📜 mySolarmodel.ipynb # Jupyter notebook for training & evaluation
┣ 📜 best_model.pth # Best performing model weights
┣ 📜 final_pv_defect_model.pth # Final trained model
┣ 📜 index.html # Frontend HTML interface
┣ 📜 requirements.txt # List of required dependencies
┣ 📜 dataset link # File containing dataset source or download link
┗ 📜 README.md # Project documentation (this file)



## ⚙️ Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/malakaminthabet/solardoc.git
   cd solardoc

    Create a virtual environment (optional but recommended)

python -m venv venv
source venv/bin/activate       # for Linux/Mac
venv\Scripts\activate          # for Windows

Install dependencies

pip install -r requirements.txt

Run the Flask web app

python app.py

Open your browser

    http://127.0.0.1:5000

    Upload a solar panel image to detect defects!

📊 Dataset

You can find the dataset link in the file:

dataset link

It contains labeled solar panel images categorized by defect type.
🧩 Model Training

If you want to train the model yourself:

jupyter notebook mySolarmodel.ipynb

Training uses transfer learning with ResNet50 and supports checkpointing, allowing you to resume from the last saved model.
🖼️ Results

SolarDoc generates:

    Training accuracy & loss curves

    Confusion matrices for model performance

    Example predictions with defect visualization boxes


🧰 Technologies Used

    Python 3.x

    PyTorch

    torchvision

    NumPy, Pandas, Matplotlib

    Flask

    HTML/CSS

    OpenCV

👩‍💻 Author

Malak Amin Thabet
📧 Contact: [your-email@example.com
]
🔗 GitHub: https://github.com/malakaminthabet
🪴 License

This project is licensed under the MIT License – feel free to use and modify it for research or development.
🌟 Acknowledgements

Special thanks to open-source datasets and the PyTorch community for their contributions to solar defect detection research.

