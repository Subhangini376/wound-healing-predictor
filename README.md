\# 🩹 Wound Healing Predictor



> A deep learning web app for wound segmentation, severity classification, and healing time prediction.





\## 📌 About The Project



This B.Tech Final Year Project uses U-Net deep learning model to:

\- Automatically \*\*segment wounds\*\* from images

\- \*\*Classify severity\*\* — Mild / Moderate / Severe / Critical

\- \*\*Predict healing time\*\* based on wound area percentage

\- Give \*\*medical advice\*\* based on severity

\- Track \*\*patient details\*\* — name, age, gender





\## 🧠 Model Performance



| Model | Dice Coefficient | IoU | Accuracy |

|---|---|---|---|

| \*\*U-Net\*\* | \*\*0.8450\*\* | \*\*0.7493\*\* | \*\*0.9960\*\* |

| U-Net++ | 0.8089 | 0.7115 | 0.9955 |





\## 📈 Healing Prediction



| Wound Area | Severity | Healing Time |

|---|---|---|

| 0 – 5% | Mild | 7–14 days |

| 5 – 15% | Moderate | 14–30 days |

| 15 – 30% | Severe | 30–60 days |

| 30%+ | Critical | 60–90+ days |







\## 🚀 How To Run



```bash

pip install tensorflow opencv-python streamlit pillow

streamlit run app\_fixed.py

🛠️ Tech Stack

TensorFlow / Keras

OpenCV

Streamlit

Python 3.10

NumPy, Pandas, Matplotlib

📊 Dataset

2208 training images + 552 test images

Image size: 256×256 pixels

👩‍💻 Author

Subhangini Patra \& Stiti Pragyan Khatua — B.Tech Final Year Project 2026



