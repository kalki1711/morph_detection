**Morphed Image Detection using CNN + Autoencoder**
   This project implements a deep learning-based application to detect morphed face images using a combination of Convolutional Neural Networks (CNNs) for feature extraction and an Autoencoder for anomaly detection. 
   It provides an interactive Streamlit web app for easy use.
*Project Overview*
   Morphing attacks on facial images can fool biometric verification systems. 
   This tool uses a CNN to extract key features from face images, then employs an autoencoder to learn the patterns of genuine faces. 
   If a test image cannot be well-reconstructed, it is flagged as "Morphed".
*Key Features*
   Load face images from nested directories
   CNN-based feature extraction
   Autoencoder for anomaly detection based on reconstruction error
   Streamlit interface for uploading and classifying new images
   Outputs prediction label, reconstruction error, and a confidence score
*How It Works*
   Data Loading: All .jpg/.jpeg/.png images are read and resized to 64x64.
   Feature Extraction: A CNN learns features of normal face images.
   Autoencoder Training: A dense autoencoder is trained to reconstruct these features.
   Prediction:The uploaded image is passed through the CNN to get features.
   Autoencoder tries to reconstruct it.
   If reconstruction error > threshold → Real Image (i.e., likely morphed).
   Otherwise → Morphed Image (i.e., looks normal).
*Run the application*
   streamlit run app.py
