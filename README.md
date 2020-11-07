## Project explanation:

Here a transfer learning solution for traffic light detection is presented. It uses [Mask Region-Based Convolutional Neural Network](https://github.com/matterport/Mask_RCNN) as it base. The model is trained on the [COCO](https://cocodataset.org/#home) data set and is capable of common object detection. It is retrained again to detect traffic lights and their current state with a specific dataset created in Skopje, North Macedonia. The model is reconfigured to mark the traffic light position, and to detect one of the three possible states: red light, green light, and transition state. In addition to the state prediction, it is also showing the certainty in the prediction from 0 to 1 as can be seen in the example below.

<img src="https://i.imgur.com/OWsCxIG.png" width="700" height="400">

## Files explanation:
### “custom.py”
>	This code is used for training a new model from pre-trained weights or resume training a model that has been trained earlier. The instructions for training are given in the first section of the code.
### “test_code.py”
>	This code is used for testing the models. Two options are available: to test the model on one image (the result is not saved) or on a collection of images (the results are saved in a folder named “aftertestImages”). The second option is suitable when this algorithm needs to be applied to a video. In my case, I used “ffmpeg” to break down the video into images.
### “customImages”
>	This folder contains the training data set (images). The resolution of the images is reduced due to faster training of the network.
### “testImages”
>	This folder contains the test data set (images) which is taken from a video file. 
### “mrcnn”
>	This folder contains all the scripts needed for creating and training the model as well as for visualization of the results.
---

** **Weights from the model trained on the COCO data set can be found on the following link: https://drive.google.com/file/d/1uO-YatiV3CtytATorxTHCwxiKCMD5XCI/view?usp=sharing.** 

** **Weights from the model trained on traffic-lights data set can be found on the following link: https://drive.google.com/file/d/1FxfY0GO_YrzRc6XtVkuKAdYPLwWfT02e/view?usp=sharing.**
