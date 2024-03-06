import requests
import base64
import cv2
import json
import os


def draw_bboxes_on_images(json_path, image_folder):
    # Read JSON file
    with open(json_path, "r") as file:
        data = json.load(file)

    # Iterate through objects in JSON
    for key, value in data.items():
        # Create path for image
        image_path = f"{image_folder}/{key}.png"

        # Read image
        image = cv2.imread(image_path)

        # Check whether image loaded
        if image is not None:
            # Iterate bounding-box information
            for bbox in value.get("bbox", []):
                # bbox in [x, y, width, height]
                x, y, w, h = bbox
                # Draw rectangular in red
                cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)

            # Save image
            cv2.imwrite(f"{image_folder}/annotated_{key}.png", image)
        else:
            print(f"Image {key}.png not found in {image_folder}")


def draw_bboxes_and_ids_on_images(json_path, image_folder):
    # Read JSON file
    with open(json_path, "r") as file:
        data = json.load(file)

    # Iterate through objects in JSON
    for key, value in data.items():
        # Create path for image
        image_path = f"{image_folder}/{key}.png"

        # Read image
        image = cv2.imread(image_path)

        # Check whether image loaded
        if image is not None:
            # Iterate bounding-box information
            for index, bbox in enumerate(value.get("bbox", [])):
                # bbox in [x, y, width, height]
                x, y, w, h = bbox
                # Draw rectangular in red
                cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)
                # Add index text
                cv2.putText(
                    image,
                    str(index + 1),
                    (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 0, 255),
                    2,
                )

            # Save image
            cv2.imwrite(f"{image_folder}/annotated_{key}.png", image)
        else:
            print(f"Image {key}.png not found in {image_folder}")


def draw_bboxes_and_index_on_images(json_path, image_folder):
    # Read JSON file
    with open(json_path, "r") as file:
        data = json.load(file)

    annotate_index = 0
    # Iterate through objects in JSON
    for key, value in data.items():
        # Create path for image
        image_path = f"{image_folder}/{key}.png"
        # Read image
        image = cv2.imread(image_path)
        # Check whether image loaded
        if image is not None:
            bbox_index = 0
            # Iterate bounding-box information
            for index, bbox in enumerate(value.get("bbox", [])):
                # bbox in [x, y, width, height]
                annotate_index = annotate_index + 1
                x, y, w, h = bbox
                # Draw rectangular in red
                cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 1)
                # Add index text
                cv2.putText(
                    image,
                    str(annotate_index),
                    (x, y + 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 0, 255),
                    2,
                )

            # Save image
            cv2.imwrite(f"{image_folder}/annotated_{key}.png", image)
        else:
            print(f"Image {key}.png not found in {image_folder}")


# Example
draw_bboxes_and_index_on_images(
    "assets/eco/eco_short_objects.json", "assets/eco/objects"
)


# OpenAI API Key
api_key = os.getenv("OPENAI_API_KEY")

# Function to encode the image


def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


# # Path to your image
# image_path = "assets/processed/objects/1.png"

# # Getting the base64 string
# base64_image = encode_image(image_path)

# headers = {"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"}

# payload = {
#     "model": "gpt-4-vision-preview",
#     "messages": [
#         {
#             "role": "user",
#             "content": [
#                 {
#                     "type": "text",
#                     "text": """I want you to work as an image semantic analyzer, in which your task is to identify the objects within the image and specifically the relationships between these objects at a semantic level. You are given an annotated image with red b-boxs and their id around the box, which has already facilitated the object detection part. The image comes from screenshots from BlackBoard Video so it's filled with illustrations and texts, you should identify the relationships between these objects.
# Considering the relationships, I only need you to to identify if there are LINEs or ARROWs DIRECTLY in the image showing the direct relationships between bboxs.
# Other levels of relationships are not considered. If there are no line connections or arrows in all the bboxs, output a [] is ok You are given:
# 1, image;
# You should return me:
# ｛
# "relation": [//The related object, should be shown together
# 2,
# 3
# ]
# }""",
#                 },
#                 {
#                     "type": "image_url",
#                     "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
#                 },
#             ],
#         }
#     ],
# }

# response = requests.post(
#     "https://api.openai.com/v1/chat/completions", headers=headers, json=payload
# )

# print(response.json())

# answer = response.json()["choices"][0]["message"]["content"]
# print(answer)
