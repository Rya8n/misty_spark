import os
import base64
from io import BytesIO
from PIL import Image
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage
from langchain_ollama import ChatOllama

def list_png_files(folder_path):
    try:
        png_files = [f for f in os.listdir(folder_path) if f.lower().endswith('.png')]
        return png_files
    except FileNotFoundError:
        print("Path does not exist.")
        return []

def convert_to_base64(pil_image):
    buffered = BytesIO()
    pil_image.save(buffered, format="PNG") 
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return img_str

def prompt_func(data):
    text = data["text"]
    images = data["images"]

    content_parts = []

    image_parts = [
        {"type": "image_url", "image_url": f"data:image/jpeg;base64,{img}"}
        for img in images
    ]

    for item in image_parts:
        content_parts.append(item)

    text_part = {"type": "text", "text": text}

    content_parts.append(text_part)
    return [HumanMessage(content=content_parts)]

def generate_observation_prompt():
    return """You are an expert at geolocation using images. 
    Step 1: Carefully analyze the images and describe:
    - Landscapes, architecture, roads, signs, vegetation, weather, or any distinctive elements.
    - Languages, text, symbols, or cultural hints.
    Do not guess the location yet. Only describe your observations."""

def generate_hypothesis_prompt(observations):
    return f"""Step 2: Based on these observations:
    {observations}
    - List possible countries or regions where this scene could be located.
    - Explain why each is plausible.
    - Identify any missing clues that could help refine the guess.
    Do not guess the location yet. Only describe your hypothesis."""

def generate_final_decision_prompt(hypothesis):
    """Prompt for step 3: Make a final location decision."""
    return f"""Step 3: Now, based on the previous reasoning:
    {hypothesis}
    - Select the most likely country and city/place.
    - Provide estimated latitude/longitude.
    - Do not provide further explanation"""

print("----------------------------------------MISTY SPARK----------------------------------------")
print("-----------------------------------------Beta v0.3-----------------------------------------")

while True:
    input("Press enter to start processing. Make sure your images are present!")
    folder_path = "footage/"
    image_files = list_png_files(folder_path)

    print(str(len(image_files))+" images loaded!")

    image_files_converted = []
    for img in image_files:
        pil_image = Image.open(folder_path+"/"+img)
        image_b64 = convert_to_base64(pil_image)
        image_files_converted.append(image_b64)

    print("Processing...")

    llm = ChatOllama(model="gemma3:12b") ### Or use your own Ollama served model

    # Observation
    observation_chain = prompt_func | llm | StrOutputParser()
    observations = observation_chain.invoke({"text": generate_observation_prompt(), "images": image_files_converted})
    print(observations)

    print(" ")

    # Hypothesis
    hypothesis_chain = prompt_func | llm | StrOutputParser()
    hypothesis = hypothesis_chain.invoke({"text": generate_hypothesis_prompt(observations), "images": image_files_converted})
    print(hypothesis)

    print(" ")

    # Final Decision
    final_decision_chain = prompt_func | llm | StrOutputParser()
    final_decision = final_decision_chain.invoke({"text": generate_final_decision_prompt(hypothesis), "images": image_files_converted})
    print(final_decision)

    print(" ")

    input("Press enter to end and start over. Simply invoke KeyboardInterrupt to quit")
