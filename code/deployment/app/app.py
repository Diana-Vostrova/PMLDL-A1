import streamlit as st
import requests
from PIL import Image
import io

st.title("Digit Recognition App")

uploaded_file = st.file_uploader("Choose an image...", type="png")

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('L')  # Convert to grayscale
    image = image.resize((28, 28))  # Resize to 28x28
    st.image(image, caption='Uploaded Image', use_column_width=True)
    
    # Convert the image to bytes
    img_byte_arr = io.BytesIO()
    image.save(img_byte_arr, format='PNG')
    img_byte_arr = img_byte_arr.getvalue()

    # Send the image to the API
    files = {'file': ('image.png', img_byte_arr, 'image/png')}
    try:
        response = requests.post("http://api:8000/predict/", files=files)
        response.raise_for_status()
        result = response.json()
        st.write(f"Predicted digit: {result['predicted_digit']}")
    except requests.exceptions.RequestException as e:
        st.error(f"Error in prediction: {str(e)}")