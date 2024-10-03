import streamlit as st
from tensorflow.keras import layers, Model
import tensorflow as tf
from tensorflow.keras.models import load_model
from rembg import remove
import numpy as np
import warnings
warnings.filterwarnings("ignore")
from PIL import Image
from tensorflow.keras.utils import get_custom_objects
import os
from keras import backend as K
from keras.saving import register_keras_serializable
from deskew import determine_skew
import cv2
from googletrans import Translator
import torch
from diffusers import StableDiffusionImg2ImgPipeline
from transformers import LlamaForCausalLM, PreTrainedTokenizerFast, pipeline
from datasets import load_dataset
from peft import LoraConfig
from trl import SFTTrainer
from transformers import pipeline
import re
import json
import gc
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    pipeline,
    logging,
)

# Language Dictionary
languages = {
    "Türkçe": "tr", 
    "Azərbaycan dili": "az",  
    "Deutsch": "de",          
    "English": "en",          
    "Français": "fr",        
    "Español": "es",         
    "Italiano": "it",        
    "Nederlands": "nl",      
    "Português": "pt",                
    "Русский": "ru",         
    "中文": "zh",             
    "日本語": "ja",           
    "한국어": "ko",           
    "عربي": "ar",          
    "हिन्दी": "hi",          
    "ภาษาไทย": "th",       
    "Tiếng Việt": "vi",      
    "فارسی": "fa",         
    "Svenska": "sv",         
    "Norsk": "no",           
    "Dansk": "da",
    "Čeština": "cs",
    "Ελληνικά": "el",   
    "Bosanski": "bs",        
    "Hrvatski": "hr",       
    "Shqip": "sq", 
    "Slovenčina": "sk",
    "Slovenščina": "sl",
    "Türkmençe": "tk", 
    "български" : "bg",
    "Кыргызча": "ky",          
    "Қазақша": "kk",           
    "Монгол": "mn",
    "Українська": "uk",
    "Cymraeg": "cy",
    "Tatarça": "tt",
    "Kiswahili": "sw",
    "Hausa": "ha",
    "አማርኛ": "am",
    "Èdè Yorùbá": "yo",
    "isiZulu": "zu",
    "chiShona": "sn",
    "isiXhosa": "xh"
}

# Translatable Texts
tr_list = ["Lyra AI E-commerce Hackathon Project", "Select Model Sharpness", "Your Product", "Your Explanation About Your Product", "Generate Image", "Generate Title and Description", "Generated Image", "Generated Title", "Generated Description"]
tr_list_tr = []

@register_keras_serializable(package='Custom', name='mse')
def custom_mse(y_true, y_pred):
    return K.mean(K.square(y_true - y_pred))

class STN(layers.Layer):
    def __init__(self, **kwargs):
        super(STN, self).__init__(**kwargs)

    def build(self, input_shape):
        self.localization = tf.keras.Sequential([
            layers.Conv2D(16, (7, 7), activation='relu', input_shape=input_shape[1:]),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Conv2D(32, (5, 5), activation='relu'),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Flatten(),
            layers.Dense(50, activation='relu'),
            layers.Dense(6, activation='linear')
        ])

    def call(self, inputs):
        theta = self.localization(inputs)
        theta = tf.reshape(theta, [-1, 2, 3])
        grid = self.get_grid(tf.shape(inputs), theta)
        return self.sampler(inputs, grid)

    def get_grid(self, input_shape, theta):
        batch_size, height, width = input_shape[0], input_shape[1], input_shape[2]
        x_coords = tf.linspace(-1.0, 1.0, width)
        y_coords = tf.linspace(-1.0, 1.0, height)
        x_grid, y_grid = tf.meshgrid(x_coords, y_coords)
        ones = tf.ones_like(x_grid)
        grid = tf.stack([x_grid, y_grid, ones], axis=-1)
        grid = tf.reshape(grid, [1, height * width, 3])
        grid = tf.tile(grid, [batch_size, 1, 1])
        grid = tf.matmul(grid, tf.transpose(theta, [0, 2, 1]))
        return grid

    def sampler(self, inputs, grid):
        shape = tf.shape(inputs)
        batch_size = shape[0]
        height = shape[1]
        width = shape[2]
        channels = shape[3]
        resized_inputs = tf.image.resize(inputs, size=(height, width))
        return resized_inputs

get_custom_objects().update({'STN': STN})

translator = Translator()

def translate_texts(language):
    return [translator.translate(text, dest=languages[language]).text for text in tr_list]

def free_up_memory():
    """Function to free up GPU memory."""
    torch.cuda.empty_cache()
    gc.collect()

def load_image_model():
    """Load the image model."""
    model = load_model("autoencoder.h5", custom_objects={'mse': custom_mse})
    return model

def load_text_model(base_model, hf_dataset):
    """Load the text model and tokenizer."""
    dataset = load_dataset(hf_dataset, split="train")
    compute_dtype = getattr(torch, "float16")

    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=False,
    )

    model = AutoModelForCausalLM.from_pretrained(base_model, quantization_config=quant_config, device_map={"": 0})
    model.config.use_cache = False
    model.config.pretraining_tp = 1
    model.low_cpu_mem_usage = True
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    peft_params = LoraConfig(
        lora_alpha=16,
        lora_dropout=0.1,
        r=64,
        bias="none",
        task_type="CAUSAL_LM"
    )

    training_params = TrainingArguments(
        output_dir="./LlamaResults",
        num_train_epochs=5,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=1,
        optim="paged_adamw_32bit",
        save_steps=25,
        logging_steps=25,
        learning_rate=2e-4,
        weight_decay=0.001,
        fp16=False,
        max_grad_norm=0.3,
        max_steps=-1,
        warmup_ratio=0.03,
        group_by_length=True,
        lr_scheduler_type="constant",
        report_to="tensorboard"
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        peft_config=peft_params,
        dataset_text_field="input",
        max_seq_length=None,
        tokenizer=tokenizer,
        args=training_params,
        packing=False,
    )
    return model, trainer, tokenizer

def process_image(input_img):
    input_img = input_img.resize((224, 224))
    input_img = np.array(input_img)
    input_img = input_img / 255.0
    input_img = np.expand_dims(input_img, axis=0)
    return input_img

def blur_level(image):
    if isinstance(image, Image.Image):
        image = np.array(image)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    laplacian = cv2.Laplacian(gray_image, cv2.CV_64F)
    variance = laplacian.var()
    return variance


# Interface
language = st.selectbox("Select Language", list(languages.keys()))
tr_list_tr = translate_texts(language)

model_choice = st.radio(tr_list_tr[0], ("Image Model", "Text Model"))
free_up_memory()

if model_choice == "Image Model":
    st.title(tr_list_tr[0])
    image_model = load_image_model()
    torch.cuda.empty_cache()

    img = st.file_uploader(tr_list_tr[2], type=["jpg", "jpeg", "png"])
    threshold = st.slider(tr_list_tr[1], min_value=50, max_value=100, value=75)
    threshold = threshold * 4

    if img:
        img = Image.open(img)
        img1 = remove(img)
        if img1.mode == 'RGBA':
            img1 = img1.convert('RGB')
        input_img = process_image(img1)
        prediction = image_model.predict(input_img)
        pred_img = np.clip(prediction[0], 0, 1) * 255
        pred_img = Image.fromarray(pred_img.astype('uint8'))
        level = blur_level(pred_img)
        st.write(level)
        if level < threshold:
            free_up_memory()
            i_pipe = StableDiffusionImg2ImgPipeline.from_pretrained("runwayml/stable-diffusion-v1-5").to("cpu")
            img = np.array(img)
            img = cv2.resize(img, (768, 768))
            img = Image.fromarray(img)
            i_prompt = "Remove the background from the image and correct the perspective of the subject to ensure a straight and clear view."
            images = i_pipe(prompt=i_prompt, image=img, strength=0.75, guidance_scale=7.5).images
            free_up_memory()
            images[0].save("output.png")
            image = Image.open("./output.png")
            st.image(image, caption=tr_list_tr[6], use_column_width=True)
            free_up_memory()
            
        else:
            st.image(pred_img, caption=tr_list_tr[6], use_column_width=True)

elif model_choice == "Text Model":
    st.title(tr_list_tr[0])
    base_model = "amd/AMD-Llama-135m"
    hf_dataset = "ahmeterdempmk/Llama-E-Commerce-Fine-Tune-Data"
    model, trainer, tokenizer = load_text_model(base_model, hf_dataset)
    torch.cuda.empty_cache()

    text = st.text_input(tr_list_tr[3])
    #text = translator.translate(text, dest="en").text
    if text and st.button(tr_list_tr[5]):
        prompt = f"""
You are extracting product title and description from given text and rewriting the description and enhancing it when necessary.
Always answer like the examples. Do not use any other keywords. Do not make up anything.

Examples:

Product Information: Rosehip Marmalade, keep it cold 
Answer: 
Recommended Title: Rosehip Marmelade
Description: You should store this delicisious roseship marmelade in cold conditions. You can use it in your breakfasts and meals.

Product Information: Blackberry jam spoils in the heat 
Answer:
Recommended Title: Blackberry Jam
Description: Please store it in cold conditions. Recommended to be consumed at breakfast. Very sweet.

Now extract the title and description from:
Product Information: 
{text}
"""
        
        pipe = pipeline(task="text-generation", model=model, tokenizer=tokenizer, max_length=1000)
        result = pipe(f"Prompt: {prompt} \n Response:")
        generated_text = result[0]['generated_text']
        #generated_text=translator.translate(text, dest=languages[language]).text
        #json_string = re.search(r'Response: (\{.*?\})', generated_text, re.DOTALL).group(1)
        #data = json.loads(json_string)
        #title = generated_text.split("Answer: Recommended Title:"
        #st.write(f"{tr_list_tr[7]}: {data['title']}")
        #st.write(f"{tr_list_tr[8]}: {data['description']}")
        st.write(generated_text)
        
free_up_memory()
