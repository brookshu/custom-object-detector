# Custom object detector

This tutorial will teach you how to manually create a dataset with Labelme and then convert it to the correct format in order to train the YOLOv5 object detection model.

## Annotations
To start, choose 2-5 classes of objects that you want to detect. You will need to choose around 30-50 images, which will be used to train the model later. It will be helpful to rename each image to something simple to keep track and easily go through them later.

The classes I chose were Person, Car, Motorcycle, and Bus. Here's an example of an image that I chose:

[image]

Next, download [Labelme](https://github.com/wkentaro/labelme/tree/main). This will be the tool used to create annotations.
