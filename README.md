# Generative-Deep-Learning-Techniques-for-Cauliflower-Disease-Detection-in-Smart-Agriculture
## 📖 Abstract

Crop diseases cause significant loss in agriculture. This system combines *CNNs* for classification and *VAEs* for latent feature extraction, improving generalization and interpretability. The model is trained on a custom cauliflower disease dataset and achieves high accuracy using advanced image preprocessing, data augmentation, and hybrid deep learning.

---

## 💡 Key Features

•⁠  ⁠🌿 Early detection of cauliflower leaf diseases
•⁠  ⁠🧠 Hybrid model: CNN for classification + VAE for feature learning
•⁠  ⁠🔄 Transfer learning + data augmentation for better accuracy
•⁠  ⁠📊 Confusion matrix, F1-score, precision, and recall-based evaluation
•⁠  ⁠🌎 Real-world agricultural dataset with environmental variations
•⁠  ⁠📈 Training and validation visualizations included

---

## 🛠️ Technologies Used

| Component       | Tools/Frameworks               |
|----------------|---------------------------------|
| Programming     | Python                         |
| Deep Learning   | TensorFlow, Keras, PyTorch     |
| Image Processing| OpenCV, NumPy                  |
| Visualization   | Matplotlib, Seaborn            |
| Hardware        | NVIDIA GPU (CUDA)              |

---

## 🌿 Dataset

•⁠  ⁠Cauliflower disease image dataset
•⁠  ⁠Classes: ⁠ Bacterial Spot Rot ⁠, ⁠ Black Rot ⁠, ⁠ Downy Mildew ⁠, ⁠ No Disease ⁠
•⁠  ⁠Images include real-field environmental noise and lighting variations
•⁠  ⁠Data preprocessing: normalization, resizing (224x224), noise filtering
•⁠  ⁠Augmentation: rotation, brightness adjustment, noise injection

---

## 🧠 Model Architecture

### 1. *CNN*
•⁠  ⁠Multiple convolutional layers + ReLU + MaxPooling
•⁠  ⁠Fully connected layers + Softmax activation for classification

### 2. *VAE*
•⁠  ⁠Encoder: Converts input to latent space
•⁠  ⁠Decoder: Reconstructs images from latent space
•⁠  ⁠Latent features used to improve classification generalization

---

## 🧪 Training & Evaluation

| Metric      | CNN Accuracy | CNN-VAE Accuracy |
|-------------|--------------|------------------|
| Accuracy    | 87%          | 86% (VAE alone)  |
| Precision   | High         | Improved recall  |
| F1-Score    | Balanced     | Balanced         |

•⁠  ⁠Training: 80%, Validation: 10%, Testing: 10%
•⁠  ⁠Batch size: 32, Epochs: 50, Learning rate: 0.001



## 📊 Visualization & Results

•⁠  ⁠📈 Training vs Validation accuracy & loss curves
•⁠  ⁠📋 Confusion Matrix: High recall in 'Bacterial Spot Rot'
•⁠  ⁠🎯 Realistic performance even under similar-looking disease conditions
•⁠  ⁠🧪 VAE aided in recognizing visual features even with class overlap

![image](https://github.com/user-attachments/assets/96799d5b-ea24-4c3a-b720-338a31cd2f1c)


![image](https://github.com/user-attachments/assets/4e7092c2-eda6-4992-b31c-b76dffa0e7f9)

 
![image](https://github.com/user-attachments/assets/a2cb80aa-1b93-4b8e-9ff8-dceaba13d787)

![image](https://github.com/user-attachments/assets/fe81fcf8-8b34-4c92-b28b-d0e51925390e)



## 🔮 Future Enhancements

•⁠  ⁠📶 IoT-based real-time disease monitoring
•⁠  ⁠🧠 Explainable AI (XAI) with Grad-CAM
•⁠  ⁠🌍 Domain adaptation for cross-regional deployment
•⁠  ⁠📦 Integration with cloud or edge platforms
•⁠  ⁠🧬 Expansion of dataset with additional crops

--
## 👥 Author

•⁠  ⁠*Natuva Bhavana* – 📧 bhavanaviswanath2@gmail.com
