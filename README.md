# PCOS-or-PCOD-Prediction

# 🩺 PCOS Classifier (CNN-based)

This project is a **Convolutional Neural Network (CNN)-based classifier** to detect **Polycystic Ovary Syndrome (PCOS)** from ultrasound images. The model is trained on a dataset containing two categories: **Normal** and **PCOS**.

The system includes:

* Dataset cleaning (removing corrupted images)
* CNN model training with **TensorFlow/Keras**
* Accuracy/Loss visualization
* Prediction of new test images

---

## 📂 Project Structure

```
PCOS-Classifier/
│── dataset/
│   ├── train/
│   │   ├── Normal/
│   │   └── PCOS/
│   └── test_images/
│
│── pcos_classifier.py   # Full training + prediction script
│── pcos_model.h5        # Saved trained model
│── README.md            # Project documentation
```

---

## ⚙️ Requirements

Install the required dependencies before running the script:

```bash
pip install tensorflow numpy matplotlib pillow
```

---

## 🚀 How to Run

### 1. Prepare Dataset

* Place your training images inside:

  ```
  dataset/train/Normal/
  dataset/train/PCOS/
  ```
* Place test images inside:

  ```
  dataset/test_images/
  ```

### 2. Train the Model

Run the script to train the CNN:

```bash
python pcos_classifier.py
```

This will:

* Clean corrupted images
* Train a CNN model
* Save the trained model as `pcos_model.h5`
* Plot accuracy & loss graphs

### 3. Predict New Images

The script will automatically predict all images inside `dataset/test_images/`.
Output will look like:

```
Predicting images in folder: dataset/test_images
image1.jpg → Normal
image2.jpg → PCOS
```

---

## 🧠 Model Architecture

* **Conv2D(32, 3x3) + MaxPooling**
* **Conv2D(64, 3x3) + MaxPooling**
* **Conv2D(128, 3x3) + MaxPooling**
* Flatten → Dense(128, relu) → Dropout(0.5)
* Output Layer: Dense(1, sigmoid)

Optimizer: **Adam**
Loss: **Binary Crossentropy**
Metrics: **Accuracy**

---

## 📊 Results

* Training & Validation Accuracy plotted per epoch
* Training & Validation Loss plotted per epoch

Example graphs:

* 📈 Accuracy improves steadily across epochs
* 📉 Loss decreases with validation stability

---

## 🔮 Future Improvements

* Use **transfer learning** (e.g., VGG16, ResNet) for higher accuracy
* Deploy model as a **Flask/Django web app**
* Integrate with **mobile app (TensorFlow Lite)** for real-time usage

---

## OUTPUT:

<img width="887" height="775" alt="Screenshot 2025-09-11 150549" src="https://github.com/user-attachments/assets/8a8093a4-9b6e-41a2-a087-c6cc22566763" />
