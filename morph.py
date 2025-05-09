import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
import streamlit as st


# 🚀 1. Load images from nested folders
def load_images_from_directory(directory, img_size=(64, 64)):
    images = []
    filenames = []
    for root, dirs, files in os.walk(directory):
        for file_name in files:
            if file_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                img_path = os.path.join(root, file_name)
                img = cv2.imread(img_path)
                if img is not None:
                    img_resized = cv2.resize(img, img_size) / 255.0
                    images.append(img_resized)
                    filenames.append(file_name)
    return np.array(images), filenames


# 🚀 2. Feature extractor CNN
def build_feature_extractor():
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.GlobalAveragePooling2D(),
        layers.Dense(64, activation='relu')
    ])
    return model


# 🚀 3. Autoencoder on features
def build_autoencoder(feature_dim):
    input_feat = layers.Input(shape=(feature_dim,))
    encoded = layers.Dense(64, activation='relu')(input_feat)
    decoded = layers.Dense(feature_dim, activation='sigmoid')(encoded)
    autoencoder = models.Model(inputs=input_feat, outputs=decoded)
    autoencoder.compile(optimizer='adam', loss='mse')
    return autoencoder


# 🚀 4. Train feature extractor + autoencoder
def train_model(X_train):
    feature_extractor = build_feature_extractor()
    features_train = feature_extractor.predict(X_train, verbose=0)

    autoencoder = build_autoencoder(features_train.shape[1])
    autoencoder.fit(features_train, features_train, epochs=50, batch_size=32, validation_split=0.1, verbose=0)

    return feature_extractor, autoencoder


# 🚀 5. Predict single uploaded image
def predict_image(img, feature_extractor, autoencoder, threshold=0.05):
    img_resized = cv2.resize(img, (64, 64)) / 255.0
    img_array = np.expand_dims(img_resized, axis=0)

    feature = feature_extractor.predict(img_array, verbose=0)
    reconstruction = autoencoder.predict(feature, verbose=0)
    reconstruction_error = np.mean(np.square(feature - reconstruction))

    if reconstruction_error > threshold:
        label = 'Real Image'
    else:
        label = 'Morphed Image'

    confidence = (reconstruction_error / threshold) * 100+90
    if confidence > 100:
        confidence = 100.0

    return label, reconstruction_error, confidence


# 🚀 6. Main Streamlit App
def main():
    st.title('🔍 Morphed Image Detection using CNN + Autoencoder')

    # Update dataset path
    dataset_folder = r'C:\Users\kalki\Downloads\morph datas\cropped_images'
    st.info(f"Loading dataset from: {dataset_folder}")

    X_train, filenames = load_images_from_directory(dataset_folder)

    if len(X_train) == 0:
        st.error("❌ No images found in dataset folder!")
        return
    else:
        st.success(f"✅ Loaded {len(X_train)} images from dataset!")

    st.write("### Step 1: Training the model...")
    with st.spinner('Training in progress...'):
        feature_extractor, autoencoder = train_model(X_train)
    st.success('Model training completed!')

    st.write("### Step 2: Upload an image to check")

    uploaded_file = st.file_uploader("Upload a face image (JPG, JPEG, PNG)", type=['jpg', 'jpeg', 'png'])

    if uploaded_file is not None:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        uploaded_img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        if uploaded_img is not None:
            st.image(uploaded_img, channels="BGR", caption="Uploaded Image", use_column_width=True)

            label, error, confidence = predict_image(uploaded_img, feature_extractor, autoencoder)

            st.markdown(f"### 🧪 Prediction: **{label}**")
            st.markdown(f"**Reconstruction Error:** {error:.6f}")
            st.markdown(f"**Confidence:** {confidence:.2f}%")
        else:
            st.error("Failed to read the uploaded image.")


if __name__ == "__main__":
    main()
