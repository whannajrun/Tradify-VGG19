# Tradify-VGG19

Welcome to Tradify Model section. This project is a Product Based Bangkit Capstone Project Team C22-PS327.

# Our Work

We're making a multi-class classification model to detect various Indonesian traditional foods.
Overall, there are already 8 classes we've trained:
1. Rendang
2. Putu Ayu
3. Bika Ambon
4. Pie Susu
5. Mochi
6. Kerak Telur
7. Pempek
8. Lapis Talas Bogor

Each class consist of 120 images that we split into 3 section: train, validation, and test.

# Tools We Used

- Python
- Tensorflow
- CutMix Generator
- Google Colab
- VGG-19 pre-trained Model

# How We Do It

Firstly we load the dataset that we've previously made. To build a model we should preprocess the data. We do several things such as rescaling (normalization), and augmentating the data twice. First using ImageGenerator from Keras and second using CutMix Generator. From ImageGenerator we costumized rotation, flip, zoom range and also shift. From CutMix Generator takes two image and label pairs to perform the augmentation.

After preprocessing we move to build the model.
To gain at least 80% accuracy we did several things, besides adding additional data augmentation we also used pretrained model, VGG-19 (because we have less training data). we choose this as the base model because it has similar layer to our first CNN model. 

we load the pretrained model & weights, freeze all layers, and then create a new model by adding our own fully connected layer & also adding dropout. we take the VGG-19 layers until the blok4_pool. For first model we get 89% accuracy and for the second model we get 82% accuracy.


# Run Our Model

- Option 1: You can simply run our .ipynb file to get a new model of yours. The accuracy will range from 80% and up to 90% with more epochs (but i don't recommend this because of the overfitting risk)
- Option 2: You can download our saved model, we already provided several formats such as .tflite, .h5, .json, and .pb. We recommend to embedded our tflite model to the application
