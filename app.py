import streamlit as st
from PIL import Image
import torch
import torchvision.transforms as transforms
import yaml
from basicsr.utils.options import  parse_options
from basicsr.models import build_model
import cv2
import numpy as np
# from basicsr.utils import ordered_yaml

def load_image(image_path):
    """ Load an image from the uploaded file and convert it to a tensor """
    file_bytes = np.asarray(bytearray(image_path.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)  # Read image from memory buffer
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
    img = torch.from_numpy(img).float().permute(2, 0, 1).unsqueeze(0) / 255.0  # (1, C, H, W) in [0, 1]
    return img

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
file_path = "/content/test_base_benchmark_x4.yml"


opt = parse_options(file_path, is_train=False)



model = build_model(opt)

# Streamlit app title and description
st.title("Super Resolution Model Demo")
st.write("Upload an image, and the model will generate a super-resolved version.")

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg","png"])

if uploaded_file is not None:
    # Load the image
    image = Image.open(uploaded_file)

    # Display the uploaded image
    st.image(image, caption='Uploaded Image', use_column_width=True)

    # Preprocess the image (resize and normalize as needed)
    preprocess = transforms.Compose([
        # transforms.Resize((256, 256)),
        transforms.Lambda(lambda img: img.convert('RGB') if img.mode == 'RGBA' else img),
        transforms.ToTensor()
    ])

    image_tensor = preprocess(image).unsqueeze(0)  # Add batch dimension
    # input_img = load_image(uploaded_file)
    image_tensor = image_tensor.to(device)

    # Perform super resolution using the model
    # model.net_g.eval()
    # with torch.no_grad():
        
    #     output = model.net_g(image_tensor)
    model.feed_data({'lq': image_tensor})  # 'lq' for low quality input
    model.test()


    # Convert the output tensor to an image
    # output_image = transforms.ToPILImage()(output.squeeze())
    output_image = transforms.ToPILImage()(model.get_current_visuals()['result'].squeeze().cpu())


    # Display the result
    st.image(output_image, caption='Super Resolved Image', use_column_width=True)