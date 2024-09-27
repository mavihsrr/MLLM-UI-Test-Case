import os
import torch
from PIL import Image
from transformers import MllamaForConditionalGeneration, AutoProcessor

DEFAULT_MODEL = "meta-llama/Llama-3.2-11B-Vision"
DEFAULT_MAX_NEW_TOKENS = 1000
DEFAULT_TEMPERATURE = 0.9
DEFAULT_TOP_P = 0.95

def load_model_and_processor(model_id: str):
    model = MllamaForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    processor = AutoProcessor.from_pretrained(model_id)
    return model, processor

def process_image(image_path: str) -> Image.Image:
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"The image file '{image_path}' does not exist.")
    return Image.open(image_path)

def generate_text_from_image(model, processor, image, prompt_text: str, temperature: float, top_p: float, max_new_tokens: int):
    inputs = processor(image, prompt_text, return_tensors="pt").to(model.device)
    output = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
    )
    generated_text = processor.decode(output[0])
    
    response_start = generated_text.find("<|begin_of_text|>") + len("<|begin_of_text|>")
    response_end = generated_text.find("<|end_of_text|>", response_start)
    return generated_text[response_start:response_end].strip()

def main(image_path: str, model_id: str = DEFAULT_MODEL, temperature: float = DEFAULT_TEMPERATURE, top_p: float = DEFAULT_TOP_P, max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS):
    model, processor = load_model_and_processor(model_id)
    image = process_image(image_path)

    ui_element = input("Which UI element would you like to generate test cases for? ")

    prompt = f"""<|image|><|begin_of_text|>Based on the image provided, please generate comprehensive UI test cases for the '{ui_element}' user interface element displayed. 
Focus on various aspects such as:

1. **Functional Test Cases**: Describe how the '{ui_element}' should behave under different conditions (e.g., button clicks, form submissions).
2. **Usability Test Cases**: Evaluate the user experience, ensuring that the '{ui_element}' is accessible and user-friendly.
3. **Visual Test Cases**: Check the alignment, colors, and visual hierarchy of the '{ui_element}'.
4. **Error Handling Test Cases**: Define how the UI should respond to invalid input or unexpected user behavior related to the '{ui_element}'.

Make sure to provide detailed steps and expected outcomes for each test case.
"""

    result = generate_text_from_image(model, processor, image, prompt, temperature, top_p, max_new_tokens)
    print(result)

image_path = "test.jpeg"
main(image_path, temperature=0.9, top_p=0.95, max_new_tokens=1000)
