import streamlit as st
from PIL import Image
import torch
import torchvision.transforms as transforms
import yaml
from basicsr.utils.options import  parse_options
from basicsr.models import build_model
# from basicsr.utils import ordered_yaml



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
file_path = "/content/drive/MyDrive/test_base_benchmark_x2.yml"


opt = parse_options(file_path, is_train=False)



model = build_model(opt)

# Streamlit app title and description
st.title("Super Resolution Model Demo")
st.write("Upload an image, and the model will generate a super-resolved version.")

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type="jpg")

if uploaded_file is not None:
    # Load the image
    image = Image.open(uploaded_file)

    # Display the uploaded image
    st.image(image, caption='Uploaded Image', use_column_width=True)

    # Preprocess the image (resize and normalize as needed)
    preprocess = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])

    image_tensor = preprocess(image).unsqueeze(0)  # Add batch dimension
    image_tensor = image_tensor.to(device)

    # Perform super resolution using the model
    model.net_g.eval()
    with torch.no_grad():
        
        output = model.net_g(image_tensor)

    # Convert the output tensor to an image
    output_image = transforms.ToPILImage()(output.squeeze())

    # Display the result
    st.image(output_image, caption='Super Resolved Image', use_column_width=True)