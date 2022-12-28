### 1. Imports and class names setup ### 
import gradio as gr
import os
import torch

from model import create_effnetb2_model
from timeit import default_timer as timer
from typing import Tuple, Dict

# Setup class names
class_names = ["Táo (Apple)", "Bơ (Avocado)", "Chuối (Banana)", "Hồng Xiêm (Sapoche)", "Quýt (Clementine)", "Dừa (Coconut)", "Thanh Long (Dragonfruit)", "Sầu Riêng (Durian)", "Nho (Grape)", "Bưởi (Jackfruit)", "Chanh (Lime)", "Nhãn (Longan)", "Vải (Lychee)", "Cam (Orange)", "Đu Đủ (Papaya)", "Dứa (Pineapple)", "Lựu (Pomegranate)", "Dâu (Strawberry)", "Dưa Hấu (Watermelon)"]

### 2. Model and transforms preparation ###

# Create EffNetB2 model
effnetb2, effnetb2_transforms = create_effnetb2_model(
    num_classes=19, # len(class_names) would also work
)

# Load saved weights
effnetb2.load_state_dict(
    torch.load(
        f="pretrained_effnetb2_feature_extractor_.pth",
        map_location=torch.device("cpu"),  # load to CPU
    )
)

### 3. Predict function ###

# Create predict function
def predict(img) -> Tuple[Dict, float]:
    """Transforms and performs a prediction on img and returns prediction and time taken.
    """
    # Start the timer
    start_time = timer()
    
    # Transform the target image and add a batch dimension
    img = effnetb2_transforms(img).unsqueeze(0)
    
    # Put model into evaluation mode and turn on inference mode
    effnetb2.eval()
    with torch.inference_mode():
        # Pass the transformed image through the model and turn the prediction logits into prediction probabilities
        pred_probs = torch.softmax(effnetb2(img), dim=1)
    
    # Create a prediction label and prediction probability dictionary for each prediction class (this is the required format for Gradio's output parameter)
    pred_labels_and_probs = {class_names[i]: float(pred_probs[0][i]) for i in range(len(class_names))}
    
    # Calculate the prediction time
    pred_time = round(timer() - start_time, 5)
    
    # Return the prediction dictionary and prediction time 
    return pred_labels_and_probs, pred_time

### 4. Gradio app ###

# Create title, description and article strings 
title = "Phân loại trái cây qua hình ảnh 🍓🍉🍌🥑🍏"
description = "Phân loại trái cây qua hình ảnh dùng EfficientNetB0 feature extractor computer vision model. Hiện tại đã phân loại được 19 loại trái cây Việt Nam gồm: Táo, Bơ, Chuối, Hồng Xiêm, Quýt, Dừa, Thanh long, Sầu riêng, Nho, Bưởi, Chanh, Nhãn, Vải, Cam, Đu Đủ, Dứa, Lựu, Dâu, Dưa hấu với tỉ lệ chính xác hơn 91%."
article = "Created by team 9: Xử lý ảnh và ứng dụng - CS406.N11.  Public Source Code: https://github.com/19522515/PhanLoaiTraiCay "

# Create examples list from "examples/" directory
example_list = [["examples/" + example] for example in os.listdir("examples")]

# Create the Gradio demo
demo = gr.Interface(fn=predict, # mapping function from input to output
                    inputs=gr.Image(type="pil"), # what are the inputs?
                    outputs=[gr.Label(num_top_classes=19, label="Predictions"), # what are the outputs?
                             gr.Number(label="Prediction time (s)")], # our fn has two outputs, therefore we have two outputs
                    theme='darkhuggingface',
                    # Create examples list from "examples/" directory
                    examples=example_list, 
                    title=title,
                    description=description,
                    article=article)

# Launch the demo!
demo.launch()
