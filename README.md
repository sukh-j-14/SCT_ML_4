# Hand Gesture Recognition System

A deep learning-based hand gesture recognition system that can accurately identify and classify different hand gestures from image data, enabling intuitive human-computer interaction and gesture-based control systems.

## 🎯 Project Overview

This project implements a Convolutional Neural Network (CNN) model to recognize 10 different hand gestures from image data. The system achieves high accuracy in classifying various hand poses, making it suitable for applications in human-computer interaction, robotics, and accessibility systems.

## 📊 Dataset

The project uses the **Hand Gesture Recognition Database** which contains:

- **10 different hand gestures**: 
  - `01_palm` - Open palm
  - `02_l` - L-shaped hand
  - `03_fist` - Closed fist
  - `04_fist_moved` - Moving fist
  - `05_thumb` - Thumb gesture
  - `06_index` - Index finger pointing
  - `07_ok` - OK sign
  - `08_palm_moved` - Moving palm
  - `09_c` - C-shaped hand
  - `10_down` - Hand pointing down

- **Data Structure**:
  - 10 subject folders (00-09)
  - Each subject contains 10 gesture folders
  - Each gesture folder contains ~200 PNG image frames
  - Total: ~20,000 images

**Dataset Source**: [Leap Gesture Recognition Dataset on Kaggle](https://www.kaggle.com/datasets/gti-upm/leapgestrecog)

## 🏗️ Architecture

The model uses a **Convolutional Neural Network** with the following architecture:

```
Input Layer: (64, 64, 3) RGB images
├── Conv2D (32 filters, 3x3) + ReLU
├── MaxPooling2D (2x2)
├── Conv2D (64 filters, 3x3) + ReLU
├── MaxPooling2D (2x2)
├── Conv2D (128 filters, 3x3) + ReLU
├── MaxPooling2D (2x2)
├── Flatten
├── Dense (128 units) + ReLU
└── Dense (10 units) + Softmax
```

**Model Parameters**: 684,490 trainable parameters

## 🚀 Features

- **High Accuracy**: Achieves 100% validation accuracy
- **Real-time Processing**: Optimized for quick inference
- **Robust Classification**: Handles 10 different hand gestures
- **Scalable Architecture**: Easy to extend for additional gestures
- **Pre-trained Model**: Includes trained model weights


## 🛠️ Installation & Setup

### Prerequisites

- Python 3.7+
- TensorFlow 2.x
- OpenCV
- NumPy
- Pandas
- Scikit-learn
- Matplotlib

### Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/sukh-j-14/SCT_ML_4.git
   cd SCT_ML_4
   ```

2. **Install dependencies**:
   ```bash
   pip install tensorflow opencv-python numpy pandas scikit-learn matplotlib
   ```

3. **Download the dataset** (if not included):
   - Download from [Kaggle Dataset](https://www.kaggle.com/datasets/gti-upm/leapgestrecog)
   

## 📖 Usage

### Training the Model

1. **Open the Jupyter notebook**:
   ```bash
   jupyter notebook handgesturerecog.ipynb
   ```

2. **Run the cells sequentially**:
   - Data loading and preprocessing
   - Model architecture definition
   - Model training
   - Evaluation and visualization

### Using the Pre-trained Model

```python
import tensorflow as tf
import cv2
import numpy as np

# Load the trained model
model = tf.keras.models.load_model('hand_gesture_model.h5')

# Load and preprocess an image
def preprocess_image(image_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (64, 64))
    img = img / 255.0
    return np.expand_dims(img, axis=0)

# Make prediction
image = preprocess_image('path_to_image.png')
prediction = model.predict(image)
gesture_class = np.argmax(prediction)
```

## 📈 Performance

### Training Results

- **Training Accuracy**: 100%
- **Validation Accuracy**: 100%
- **Training Loss**: ~0.000007
- **Validation Loss**: ~0.0003

### Model Performance

The model demonstrates excellent performance with:
- Perfect classification accuracy on the validation set
- Fast convergence (10 epochs)
- No overfitting observed
- Robust feature extraction

## 🔧 Model Details

### Data Preprocessing

1. **Image Loading**: PNG images loaded using OpenCV
2. **Resizing**: Images resized to 64x64 pixels
3. **Normalization**: Pixel values normalized to [0, 1] range
4. **Data Augmentation**: Random shuffling for training

### Training Configuration

- **Optimizer**: Adam
- **Loss Function**: Sparse Categorical Crossentropy
- **Batch Size**: 32
- **Epochs**: 10
- **Validation Split**: 20%

### Gesture Classes

| Class | Gesture | Description |
|-------|---------|-------------|
| 0 | 01_palm | Open palm gesture |
| 1 | 02_l | L-shaped hand |
| 2 | 03_fist | Closed fist |
| 3 | 04_fist_moved | Moving fist |
| 4 | 05_thumb | Thumb gesture |
| 5 | 06_index | Index finger pointing |
| 6 | 07_ok | OK sign |
| 7 | 08_palm_moved | Moving palm |
| 8 | 09_c | C-shaped hand |
| 9 | 10_down | Hand pointing down |

## 🎯 Applications

This hand gesture recognition system can be used in various applications:

- **Human-Computer Interaction**: Control applications with hand gestures
- **Robotics**: Gesture-based robot control
- **Accessibility**: Assistive technology for differently-abled users
- **Gaming**: Gesture-based game controls
- **Smart Home**: Gesture-based home automation
- **Virtual Reality**: Hand tracking in VR environments

## 🔮 Future Enhancements

- **Real-time Video Processing**: Extend to video streams
- **Additional Gestures**: Support for more hand gestures
- **Multi-hand Recognition**: Detect and classify multiple hands
- **Gesture Sequences**: Recognize gesture patterns over time
- **Mobile Deployment**: Optimize for mobile devices
- **Web Interface**: Create a web-based demo

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 👨‍💻 Author

This project was developed as part of a machine learning course focusing on computer vision and deep learning applications.

## 🙏 Acknowledgments

- **Dataset Source**: [Leap Gesture Recognition Dataset](https://www.kaggle.com/datasets/gti-upm/leapgestrecog)
- **Repository**: [GitHub Repository](https://github.com/sukh-j-14/SCT_ML_4)
- **Framework**: TensorFlow/Keras
- **Computer Vision**: OpenCV
- **Data Processing**: NumPy, Pandas, Scikit-learn

---

**Note**: This model is trained on a specific dataset and may require retraining or fine-tuning for different use cases or environments. 