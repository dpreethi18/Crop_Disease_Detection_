import os
import json
from PIL import Image, ImageEnhance, ImageOps, ImageFilter
import numpy as np
import tensorflow as tf
import streamlit as st
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Load the pre-trained model
model_path = r"C:\Users\hp\Downloads\leafdataset\saved_models\model.keras"
model = tf.keras.models.load_model(model_path)

# loading the class names
working_dir = os.path.dirname(os.path.abspath(__file__))
class_indices = json.load(open(f"{working_dir}/class_indices.json"))


# Function to Load and Preprocess the Image using Pillow
def load_and_preprocess_image(image_path, target_size=(224, 224), augmentation_options=None):
    img = Image.open(image_path)
    
    # Apply augmentation options if provided
    if augmentation_options:
        img = apply_augmentation(img, augmentation_options)
    
    img = img.resize(target_size)
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array.astype('float32') / 255.
    return img, img_array


# Function to apply image augmentation
def apply_augmentation(image, augmentation_options):
    if 'brightness_factor' in augmentation_options:
        brightness_factor = augmentation_options['brightness_factor']
        enhancer = ImageEnhance.Brightness(image)
        image = enhancer.enhance(brightness_factor)
        
    if 'contrast_factor' in augmentation_options:
        contrast_factor = augmentation_options['contrast_factor']
        enhancer = ImageEnhance.Contrast(image)
        image = enhancer.enhance(contrast_factor)
    
    if 'horizontal_flip' in augmentation_options and augmentation_options['horizontal_flip']:
        image = image.transpose(Image.FLIP_LEFT_RIGHT)
        
    if 'vertical_flip' in augmentation_options and augmentation_options['vertical_flip']:
        image = image.transpose(Image.FLIP_TOP_BOTTOM)
        
    if 'rotation_angle' in augmentation_options:
        rotation_angle = augmentation_options['rotation_angle']
        image = image.rotate(rotation_angle)
    
    if 'random_noise' in augmentation_options and augmentation_options['random_noise']:
        image = image.filter(ImageFilter.GaussianBlur(radius=1))
    
    return image


# Function to Predict the Class of an Image
def predict_image_class(model, image_array):
    predictions = model.predict(image_array)
    predicted_class_index = np.argmax(predictions, axis=1)[0]
    predicted_class_name = class_indices[str(predicted_class_index)]
    return predicted_class_name, predictions


# Function to visualize activation maps
def visualize_activation_maps(model, image_array):
    activation_maps = []
    for layer in model.layers:
        if 'conv' in layer.name:
            activation_model = tf.keras.Model(inputs=model.inputs, outputs=layer.output)
            activations = activation_model.predict(image_array)
            activation_map = np.mean(activations, axis=-1)
            activation_map = np.maximum(0, activation_map)
            activation_map /= np.max(activation_map)
            activation_maps.append(activation_map)
    return activation_maps


# Function to generate heatmap visualization
def generate_heatmap(activation_map):
    fig, ax = plt.subplots()
    heatmap = ax.imshow(activation_map[0], cmap='viridis')
    ax.axis('off')
    return fig, ax


# Streamlit App
st.title('Plant Disease Classifier')

uploaded_images = st.file_uploader("Upload images...", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

if uploaded_images:
    # Data augmentation options
    augmentation_options = {
        'brightness_factor': st.slider('Brightness Factor', min_value=0.5, max_value=1.5, value=1.0, step=0.1),
        'contrast_factor': st.slider('Contrast Factor', min_value=0.5, max_value=1.5, value=1.0, step=0.1),
        'horizontal_flip': st.checkbox('Horizontal Flip', value=False),
        'vertical_flip': st.checkbox('Vertical Flip', value=False),
        'rotation_angle': st.slider('Rotation Angle', min_value=0, max_value=360, value=0, step=1),
        'random_noise': st.checkbox('Add Random Noise', value=False)
    }

    for uploaded_image in uploaded_images:
        image, image_array = load_and_preprocess_image(uploaded_image, augmentation_options=augmentation_options)

        col1, col2 = st.columns(2)

        with col1:
            resized_img = image.resize((150, 150))
            st.image(resized_img)

        with col2:
            if st.button(f'Classify {uploaded_image.name}'):
                prediction, _ = predict_image_class(model, image_array)
                st.success(f'Prediction: {str(prediction)}')

                # Visualize activation maps
                activation_maps = visualize_activation_maps(model, image_array)
                for i, activation_map in enumerate(activation_maps):
                    st.subheader(f'Activation Map for Convolutional Layer {i+1}')
                    fig, ax = generate_heatmap(activation_map)
                    st.pyplot(fig)
