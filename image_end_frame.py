import requests
import base64
import cv2
import json
import os
import random


def draw_bboxes_and_ids_on_final_images(
    json_path, image_folder, image_path, random_color
):
    # Read JSON file
    with open(json_path, "r") as file:
        data = json.load(file)

    # Read the final frame image
    image = cv2.imread(image_path)
    # Iterate through JSON

    # Check whether image loaded
    if image is not None:
        annotate_index = 0
        for key, value in data.items():
            # iterate bounding-box for each object
            annotate_index = annotate_index + 1
            bbox_index = -1
            for index, bbox in enumerate(value.get("bbox", [])):
                # bbox in [x, y, width, height]
                bbox_index = bbox_index + 1
                x, y, w, h = bbox
                color_r = 255
                color_g = 0
                color_b = 0
                if random_color:
                    color_r = random.randrange(0, 256)
                    color_g = random.randrange(0, 256)
                    color_b = random.randrange(0, 256)
                # Draw rectangular boxes in red
                cv2.rectangle(
                    image, (x, y), (x + w, y + h), (color_b, color_g, color_r), 1
                )
                # Add index text
                cv2.putText(
                    image,
                    str(annotate_index) + "_" + str(bbox_index),
                    (x, y + 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (color_b, color_g, color_r),
                    2,
                )
    else:
        print(f"Image {key}.png not found in {image_folder}")
    # save image
    cv2.imwrite(f"{image_folder}/final_frame.png", image)


def draw_bboxes_and_ids_on_images(json_path, image_folder, image_path, count):
    # Read JSON file
    with open(json_path, "r") as file:
        data = json.load(file)

    annotate_index = 0
    # Read image
    image = cv2.imread(image_path)
    # Iterate through objects in JSON
    for key, value in data.items():
        # Check whether image loaded
        if annotate_index % count == 0:
            image = cv2.imread(image_path)
        if image is not None:
            # Iterate bounding-box information
            annotate_index = annotate_index + 10
            for index, bbox in enumerate(value.get("bbox", [])):
                # bbox in [x, y, width, height]
                x, y, w, h = bbox
                # Draw rectangular in red
                cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 1)
                # Add index text
                cv2.putText(
                    image,
                    str(annotate_index),
                    (x, y + 25),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 0, 255),
                    1,
                )
        else:
            print(f"Image {key}.png not found in {image_folder}")
        if annotate_index % count == 0:
            cv2.imwrite(f"{image_folder}/annotated_{key}.png", image)
    cv2.imwrite(f"{image_folder}/annotated_{key}.png", image)


# # Example
# draw_bboxes_and_ids_on_images(
#     "assets/json/chem_short_objects.json",
#     "assets/processed/objects",
#     "assets/processed/images/5850.png",
#     4,
# )

draw_bboxes_and_ids_on_final_images(
    "assets/math/math_short_objects.json",
    "assets/math/objects",
    "assets/math/images/6570.png",
    False,
)
