# Import required modules
import requests
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration

# Load BLIP processor and model (large version for better understanding + VQA)
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large")

# Load image from URL
img_url = "https://storage.googleapis.com/sfr-vision-language-research/BLIP/demo.jpg"
image = Image.open(requests.get(img_url, stream=True).raw).convert("RGB")

# Define question about the image
question = "What is in the image?"

# Prepare both image + question as input
inputs = processor(image, question, return_tensors="pt")

# Generate answer
outputs = model.generate(**inputs)

# Decode output into readable text
answer = processor.decode(outputs[0], skip_special_tokens=True)

# Print final answer
print(answer)
