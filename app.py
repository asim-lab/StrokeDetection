import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import gdown  # Add this import

# Set the page title and icon
st.set_page_config(page_title="Stroke Detection", page_icon="🧠", layout="wide")

# Header with background color and logo
st.markdown(
    """
    <style>
    .header {
        padding: 20px;
        text-align: center;
        background-color: #4CAF50;
        color: white;
        font-size: 34px;
        border-radius: 10px;
    }
    </style>
    <div class="header">
        <span>🧠 AI-Powered Stroke Diagnosis: A Web Application for Enhanced Medical Imaging</span>
    </div>
    """, 
    unsafe_allow_html=True
)



# Goals Section
st.markdown("## Project Goals")
st.markdown(
    """

1. **Accurate Stroke Classification**
   - Develop an AI model-based web app to classify medical images into categories such as ischemic stroke, hemorrhagic stroke, and normal.
   - Provide reliable diagnostic support to healthcare professionals.

2. **Improved Treatment Planning**
   - Enable neurologists to use the AI model's diagnosis results to devise appropriate treatment plans for stroke patients.
   - Enhance patient outcomes and recovery processes.

3. **Efficient Healthcare Delivery**
   - Reduce unnecessary hospital visits by offering remote diagnostic capabilities.
   - Improve healthcare efficiency and minimize the burden on both patients and healthcare facilities.

4. **Cost Reduction and Accessibility**
   - Lower overall healthcare costs by reducing the need for frequent in-person consultations.
   - Make stroke diagnosis more accessible and affordable, especially in remote areas.
   """
)

# Sidebar with instructions
st.sidebar.title("Upload an Image to Check the Stroke by AI Model")
st.sidebar.info(
    """
    - Upload an image in PNG format.
    - The model will classify the image into either 'Stroke' or 'Normal'.
    """
)

@st.cache_resource
def load_model():
    # Define the Google Drive file ID
    file_id = '11NAOliP_stgzo2mAix60AR-FcmukspLT'  # Replace with your actual file ID
    url = f'https://drive.google.com/uc?id={file_id}'

    # Download the model from Google Drive
    output = 'stroke_model_trained.h5'
    gdown.download(url, output, quiet=False)
    
    # Load the downloaded model
    model = tf.keras.models.load_model(output)
    return model

def predict_class(image, model):
    image = tf.cast(image, tf.float32)
    image = tf.image.resize(image, [180, 180])
    image = np.expand_dims(image, axis=0)
    prediction = model.predict(image)
    return prediction

model = load_model()

# Main section for file upload
st.markdown("### Upload your image below:")
file = st.file_uploader("Choose an image...", type=["png"])

if file is None:
    st.markdown("### Waiting for an image to be uploaded...", unsafe_allow_html=True)
else:
    st.markdown("### Running inference...")
    
    test_image = Image.open(file)
    
    # Display uploaded image with border and shadow
    st.markdown(
        """
        <style>
        .image-container {
            border: 2px solid #4CAF50;
            border-radius: 10px;
            padding: 10px;
            box-shadow: 2px 2px 5px rgba(0, 0, 0, 0.2);
        }
        </style>
        <div class="image-container">
        </div>
        """, unsafe_allow_html=True
    )
    st.image(test_image, caption="Input Image", width=400)
    
    # Prediction
    pred = predict_class(np.asarray(test_image), model)
    class_names = ['Stroke', 'Normal']
    result = class_names[np.argmax(pred)]
    
    # Display result with custom styling
    st.markdown(
        f"""
        <div style="text-align:center;">
            <h3>The image is classified as:</h3>
            <span style="font-size:24px; color:green;">{result}</span>
        </div>
        """, unsafe_allow_html=True
    )

# Footer with copyright info
st.markdown(
    """
    <style>
    .footer {
        padding: 10px;
        text-align: center;
        background-color: #4CAF50;
        color: white;
        font-size: 12px;
        border-radius: 12px;
        margin-top: 20px;
    }
    </style>
    <div class="footer">
        © 2024 Center of Excellence in Artificial Intelligence, Machine Learning and Smart Grid Technology, Department of Electrical Engineering, Faculty of Engineering, Chulalongkorn University, Bangkok 10330, Thailand. All rights reserved.
    </div>
    """, unsafe_allow_html=True
)
