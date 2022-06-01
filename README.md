# Whale-Identification

This repository contains code for training a Neural Network to identify individual whales and dolphins from their back-fins.
![dorsal-fin-crops](https://user-images.githubusercontent.com/33522459/164789213-3e6ed9ef-43d4-4714-85a5-c98b42cbb720.png)

To train a Neural Network:
Step 1: Set up an AWS account and create an S3 bucket. Store the files in this repository in your S3 bucket.
Step 2: Pull train.sh into your desired computer using AWS CLI.
Step 3: Run train.sh to train your model. When training is completed, your trained models should be in the "models/" folder of your bucket.

You can choose a different model backbone by changing the model name within the whale_train.py script.
The model name must be among the available models in the Pytorch Timm library. 
  
# Arc Margin Distance

The model identifies individuals by creating a vector embedding for each image, then comparing how closely the embedding matches to all known individuals. 
By trying to reduce the angle between embedded vectors of images of the same individuals, embeddings of the same individuals are grouped tightly, and embeddings of different individuals are cleanly separated. 

![arcfaceloss](https://user-images.githubusercontent.com/33522459/164778056-9dcd3f6d-03f6-4374-a17a-ac8687b81c4a.png)

# K-Nearest Neighbors

A K-Nearest Neighbors model is trained using the vector embeddings pproduced from the neural network to identify groups formed by these vectors based on the number of other vectors closest to it. When a new picture of an individual is passed through the neural network, the vector embedding output is inputted into the KNN model and will find which group this individual belongs to.
