# LSVP Tool

LSVP Tool is an ongoing project that aims to help users convert existing online educational videos (digital blackboard-based videos) into the LSVP (Layered Serial Visual Presentation) style. This style that has been found to be useful for viewing dynamic information on OHMDs on the go. The application involves Python for backend and Javascript on the front end.

## Installation

1. Install miniconda, create a new enivronment and activate it.

2. Install necessary packages using below command

```bash
conda env create -f env.yml
```

Note: Some packages may need to be installed using pip when you run the code.

3. Clone the github folder and cd into it.

4. Download the test videos from <a href="https://drive.google.com/drive/folders/1MXc0LxuIU0qy6bFnreKzndDfQGkO_Xk2?usp=sharing">google drive</a> and place it them in assets/test folder.

## Usage

To run the object extraction pipeline

1. Create folders called /assets/test/images and /assets/test/trans_images.

2. Set the video_name in main.py and choose the correct cursor_template (check the video to see what the cursor looks like).

- eco.mp4 --> cursor_eco
- bio.mp4 and chem.mp4 --> cursor_bio
- phy.mp4 and math.mp4 --> cursor_hand

3. Run the below command

```bash
python revised_main.py
```

Note: You need to run the load_video function only once for a video. If you choose a new video for processing, empty the /images and /trans_images folder before running the code again.

To be updated: Running the outline.py file


The first image will be the whole picture and the different bounding-boxes on it, which will help you to understand the overall

4. Prompt for GPT:
### Prompt for clustering
- To start: 
```
I was hoping you could work as an image semantic analyzer, in which your task is to identify the objects within a series of the same images that have different annotations to point out objects in it and specifically the relationships between these objects at a semantic level. You are given a series of annotated images, the images come from screenshots from BlackBoard Video so it's filled with illustrations. We will do this step by step, I will send you the overview image first so that you can understand the overall semantic relationship within it. Then the images will be sent by groups.

Please keep with the following general rules for the procedure:
1. You are given a series of annotated images with red bounding boxes and their red index at the top-left corner of the box, try to identify these red indexes to find out the target objects.
2. Please do not try to identify any text within the red boxes, except the red index at the top-left corner of the box.
3. You should consider all images as a whole scene and identify the relationships between these objects in all images as a series.
4. When you receive a new image, please consider it as a whole body with previous images and re-identify the relationships within them.

Furthermore, to successfully cluster the objects, the following are specific clustering rules that you need to keep with:
1. Please cluster these objects based on their logic relations, like lines or arrows that connect different objects or in the same math equation.
2. Please cluster these objects based on their color and distance, the objects with the same color and shorter distances are more likely to have a high possibility in one cluster.
3. Please cluster these objects based on their index, which is the sequence in which they appear. You should only cluster objects that continuously index together, for example, [1, 2, 3] could be a cluster but [1, 2, 4] cannot be a cluster.
4. Please do not cluster objects based on their shapes (e.g. lines shouldn't be clustered together) and the text in them.
5. Please do not cluster objects with a large distance together, that is more important if there are nearer object that could be include.

You should return me in the format of JSON as shown below:
｛
"relation": 
//The first cluster
[//The related object indicated, should be shown together
2, //index annotated at the top left corner of red boxes
3
],

//The second cluster
[4, 5, 6],

//More clusters, we are expecting to have about 6 clusters in total
[...]
}
```

- Provide overview image for GPT
```
// Upload overview image
Here is an overview of the whole image and different objects. Please do not directly use it for clustering as there are obstacles between objects and boxes within it. The topic of this video is a lecture about kidneys, you can also use this information for semantic clustering. After this image, I will also send you the break-down images for each object, please then try to cluster them and refer back to this overview image for their overall relationships.
```

- To input images
```
//Upload images
Here are the first 6 images, there will be 15 images in total, the other images will be sent later. Please try to identify the relationships within these images based on the rules described above. Please notice that each box should be included into an cluster and could only appear in one cluster, the identified cluster should only include continuous index like [1, 2, 3] and shouldn't include any discontinuous index like [1, 2, 5]:
```

- If there are more than 10 images
```
//Upload other images
// If not finished use: Here are the next 6 images, there are still 3 images remaining
Thanks, you are doing great. Now I will send you the last 5 images, please consider the objects within these new images together with the previous images to identify the relationships within them again, please notice that the objects in the new image could form a new cluster with the previous objects and can also join old clusters. We expect that there will be about 6 clusters in total, the identified cluster should only include continious index like [1, 2, 3] and shouldn't include any discontinuous index like [1, 2, 5]:
```


### Prompt for lines and arrows
- To start: 
```
I want you to work as an image semantic analyzer, in which your task is to identify the objects within the image and specifically the relationships between these objects at a semantic level. You are given an annotated image with red b-boxs and their id around the box, which has already facilitated the object detection part. The image comes from screenshots from BlackBoard Video so it's filled with illustrations and texts, you should identify the relationships between these objects.
Considering the relationships, I only need you to identify if there are LINEs or ARROWs DIRECTLY in the image showing the direct relationships between bboxs.
Other levels of relationships are not considered. If there are no line connections or arrows in all the bboxs, output a [] is ok.
You should return me:
｛
"connections": [//The related object, should be shown together
2,
3
]
}


```

- Provide overview image for GPT
```
// Upload overview image
Here is an overview of the whole image and different objects. Please do not directly use it for clustering as there are obstacles between objects and boxes within it. The topic of this video is a lecture about kidneys, you can also use this information for semantic clustering. After this image, I will also send you the break-down images for each object, please then try to cluster them and refer back to this overview image for their overall relationships.
```

- To input images
```
//Upload images
Here are the first 6 images, there will be 15 images in total, the other images will be sent later.
```

- If there are more than 10 images
```
//Upload other images
// If not finished use: Here are the next 6 images, there are still 3 images remaining
Thanks, you are doing great. Now I will send you the last 5 images.
```