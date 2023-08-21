# Custom object detector

This tutorial will teach you how to manually create a dataset with Labelme and then convert it to the correct format in order to train the YOLOv5 object detection model.

## Annotations
To start, choose 2-5 classes of objects that you want to detect. You will need to choose around 30-50 images, which will be used to train the model later. It will be helpful to rename each image to something simple to keep track and easily go through them later.

The classes I chose were Person, Car, Motorcycle, and Bus. Here's an example of an image that I chose:

[image]

Next, download [Labelme](https://github.com/wkentaro/labelme/tree/main). This will be the tool used to create annotations.

In each image, create polygons around each object in the photo. It can be done manually with the polygon tool, but the AI-polygon tool can help speed things up.

[image]

After creating the polygon, the following window will pop up. Label them as the class name. Two polygons having the same class ID will count as the same object, so this can be used if another object or the background splits the object in half. Otherwise, do not enter a class ID.

After saving each image, a .json file should pop up in the same folder as your images with information on the polygons you just created.

## Converting Labelme to JSON/COCO format

To start, we'll convert from the Labelme format to JSON format

https://github.com/brookshu/custom-object-detector/blob/f2b5493bfc3f14f3a04595154d666bdab83e979c/convert_labelme.py

This code takes three arguments with `argparse`: a path to the directory with .json files from Labelme, a path to an output directory, and the path to [labels.txt](), which has a list of the classes you are using. 

See labels.txt too and make ur own

A couple changes were made from the [original](https://github.com/wkentaro/labelme/blob/main/examples/instance_segmentation/labelme2coco.py). ?

Output should be a folder (see my [example](https://github.com/brookshu/custom-object-detector/tree/9725705cff8a98b8b320a7b6a507d6602bbb3dab/annotations)) with annotations.json file and Visualization and JPEGImages directory.

## Converting from JSON/COCO to YOLO format

Then, convert from the JSON




