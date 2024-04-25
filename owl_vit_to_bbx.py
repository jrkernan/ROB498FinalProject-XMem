<<<<<<< HEAD
import requests
from PIL import Image, ImageFont, ImageDraw, ImageEnhance
import torch
from transformers import OwlViTProcessor, OwlViTForObjectDetection


## arguments: 
# image path: path to first image of video
# text prompt: vector of a string asking for a photo of the object to track. EX: ["a photo of a cat"] 
# k: how many of said object instances do you want to track? default is 1
def get_bbx(image_path, text_prompt, k=1):

    processor = OwlViTProcessor.from_pretrained("google/owlvit-base-patch32")
    model = OwlViTForObjectDetection.from_pretrained("google/owlvit-base-patch32")

    image = Image.open(image_path)
    # draw = ImageDraw.Draw(image)
    texts = [text_prompt]
    inputs = processor(text=texts, images=image, return_tensors="pt")
    outputs = model(**inputs)

    # Target image sizes (height, width) to rescale box predictions [batch_size, 2]
    target_sizes = torch.Tensor([image.size[::-1]])
    # Convert outputs (bounding boxes and class logits) to Pascal VOC format (xmin, ymin, xmax, ymax)
    results = processor.post_process_object_detection(outputs=outputs, target_sizes=target_sizes, threshold=0.1)
    i = 0  # Retrieve predictions for the first image for the corresponding text queries
    text = texts[i]
    boxes, scores, labels = results[i]["boxes"], results[i]["scores"], results[i]["labels"]
    
    result= []
    for box, score, label in zip(boxes, scores, labels):
        box = [round(i, 2) for i in box.tolist()]
        confidence = round(score.item(), 3)
        result.append([box,confidence])
        # print(f"Detected {text[label]} with confidence {round(score.item(), 3)} at location {box}")
    result.sort(key=lambda x: x[1])
    # draw.rectangle(result[0][0], fill=(255, 0, 0),outline=(0, 0, 0))
    # image.save("output.jpg","JPEG")
    result = [row[0] for row in result]
    return result[:k]



=======
import requests
from PIL import Image, ImageFont, ImageDraw, ImageEnhance
import torch
from transformers import OwlViTProcessor, OwlViTForObjectDetection


## arguments: 
# image path: path to first image of video
# text prompt: vector of a string asking for a photo of the object to track. EX: ["a photo of a cat"] 
# k: how many of said object instances do you want to track? default is 1
def get_bbx(image_path, text_prompt, k=1):

    processor = OwlViTProcessor.from_pretrained("google/owlvit-base-patch32")
    model = OwlViTForObjectDetection.from_pretrained("google/owlvit-base-patch32")

    image = Image.open(image_path)
    # draw = ImageDraw.Draw(image)
    texts = [text_prompt]
    inputs = processor(text=texts, images=image, return_tensors="pt")
    outputs = model(**inputs)

    # Target image sizes (height, width) to rescale box predictions [batch_size, 2]
    target_sizes = torch.Tensor([image.size[::-1]])
    # Convert outputs (bounding boxes and class logits) to Pascal VOC format (xmin, ymin, xmax, ymax)
    results = processor.post_process_object_detection(outputs=outputs, target_sizes=target_sizes, threshold=0.1)
    i = 0  # Retrieve predictions for the first image for the corresponding text queries
    text = texts[i]
    boxes, scores, labels = results[i]["boxes"], results[i]["scores"], results[i]["labels"]
    
    result= []
    for box, score, label in zip(boxes, scores, labels):
        box = [round(i, 2) for i in box.tolist()]
        confidence = round(score.item(), 3)
        result.append([box,confidence])
        # print(f"Detected {text[label]} with confidence {round(score.item(), 3)} at location {box}")
    result.sort(key=lambda x: x[1])
    # draw.rectangle(result[0][0], fill=(255, 0, 0),outline=(0, 0, 0))
    # image.save("output.jpg","JPEG")
    result = [row[0] for row in result]
    return result[:k]



>>>>>>> fba9a3707e2043a43a5173056942b24324ab4bb2
