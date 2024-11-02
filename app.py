import streamlit as st
import cv2                          # Add OpenCV for preprocessing and postprocessing
from PIL import Image, ImageFilter
import numpy as np
from streamlit_drawable_canvas import st_canvas
import torch
import torchvision.transforms as transforms
import yaml
from basicsr.utils.options import parse_options
from basicsr.models import build_model
import io

# Set page layout to wide
st.set_page_config(layout="wide")

# Device setup for PyTorch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

file_path = "/content/test_base_benchmark_x2.yml"

# Load the selected model configuration
opt = parse_options(file_path, is_train=False)
model = build_model(opt)

def preprocess_image(image):
    # Directly return the original image without any preprocessing
    return image

def advanced_postprocess_image(image):
    # Convert image to PIL format (if not already) and apply a mild sharpening filter
    sharpened_image = image.filter(ImageFilter.UnsharpMask(radius=1, percent=100, threshold=1))
    return sharpened_image

# Streamlit app title and description
# st.title(f"Image Upscaling with {model_selection}")
st.title(f"Generating High Resolution Zoom-In for Images using LMLT-Base-x2")

# File uploader
uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "png", "jpeg"])

# Display the uploaded image first, then provide options
if uploaded_file is not None:
    # Load and display the image
    image = Image.open(uploaded_file)
    img_width, img_height = image.size
    st.image(image, caption="Uploaded Image", use_column_width=False)

    # Provide two options for upscaling
    # option = st.radio("Choose Upscaling Option:", ("Upscale Entire Image", "Upscale Selected Region"))
    option = st.selectbox("Choose Upscaling Option", ["Upscale Entire Image", "Upscale Selected Region"])

    # Upscale Entire Image Option
    if option == "Upscale Entire Image":
        if st.button("Upscale Entire Image"):
            try:
                # Preprocess the image to reduce noise
                preprocessed_image = preprocess_image(image)

                # Convert preprocessed image to tensor for model input
                preprocess = transforms.Compose([
                    transforms.Lambda(lambda img: img.convert('RGB') if img.mode == 'RGBA' else img),
                    transforms.ToTensor()])
                image_tensor = preprocess(preprocessed_image).unsqueeze(0).to(device)

                # Feed the image to the model and run upscaling
                model.feed_data({'lq': image_tensor})
                model.test()

                # Get the output from the model
                output_image_tensor = model.get_current_visuals()['result'].squeeze().cpu()
                output_image = transforms.ToPILImage()(output_image_tensor)

                # Postprocess the output image to reduce color patches
                #output_image = postprocess_image(output_image)

                # Apply advanced postprocessing to reduce color artifacts
                # output_image = advanced_postprocess_image(output_image)                  # --- Last Edit --- Good Results

            except RuntimeError:
                st.write("##### ▲ An error occurred while processing the image. Please try with a smaller image.")

            else:
                #display_width = calculate_display_dimensions_from_height(img_width, img_height, 2, 300)
                st.image(output_image, caption="Upscaled Entire Image", width= 2 * img_width)
                buf = io.BytesIO()
                output_image.save(buf, format='PNG')
                st.download_button("Download Upscaled Image", buf.getvalue(), file_name="upscaled_image.png", mime="image/png")

    # Upscale Selected Region Option
    elif option == "Upscale Selected Region":
        st.subheader("Select Region of Interest by Drawing a Rectangle")

        # Interactive canvas for region selection
        canvas_result = st_canvas(
            fill_color="rgba(255, 165, 0, 0.3)",  # Choose a fill color for selection
            stroke_width=3,
            background_image=image,
            update_streamlit=True,
            height=img_height,
            width=img_width,
            drawing_mode="rect",  # Enable rectangle drawing
            key="canvas"
        )

        if canvas_result.json_data is not None:
            # Get bounding box of drawn rectangle
            objects = canvas_result.json_data["objects"]
            if objects:
                # Take the first drawn rectangle as the region of interest
                obj = objects[0]
                left = obj["left"]
                top = obj["top"]
                width = obj["width"]
                height = obj["height"]

                # Convert to integer coordinates
                left, top, width, height = map(int, [left, top, width, height])

                # Extract and display the region of interest (ROI)
                roi = np.array(image)[top:top + height, left:left + width, :]
                col1, col2 = st.columns(2)
                with col1:
                    st.image(roi, caption="Selected Region of Interest", width = img_width)

                if st.button("Upscale Selected Region"):
                    try:
                        # Convert Numpy to  Image
                        roi_image = Image.fromarray(roi)

                        # Preprocess the image to reduce noise
                        preprocessed_image = preprocess_image(roi_image)

                        preprocess = transforms.Compose([
                            transforms.Lambda(lambda img: img.convert('RGB') if img.mode == 'RGBA' else img),
                            transforms.ToTensor()])
                        roi_tensor = preprocess(preprocessed_image).unsqueeze(0).to(device)

                        # Feed the ROI to the model
                        model.feed_data({'lq': roi_tensor})
                        model.test()

                        # Convert the output tensor to an image
                        output_image = transforms.ToPILImage()(model.get_current_visuals()['result'].squeeze().cpu())

                        # Apply advanced postprocessing to reduce color artifacts
                        output_image = advanced_postprocess_image(output_image)  

                    except RuntimeError:
                        # st.error("An issue occurred while processing the image. Please try selecting a smaller region or use a different image.")
                        st.write("##### ▲ An error occurred while processing the image. Please try selecting a smaller region or use a different image.")

                    else:
                        # Display the upscaled region and provide download option
                        with col2:
                            st.image(output_image, caption="Upscaled Region", width = img_width)
                            buf = io.BytesIO()
                            output_image.save(buf, format='PNG')
                            st.download_button("Download Upscaled Region", buf.getvalue(), file_name="upscaled_region.png", mime="image/png")
