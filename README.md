# Whale-Identification

This repository contains code for training a Neural Network to identify individual whales and dolphins from their back-fins.
![dorsal-fin-crops](https://user-images.githubusercontent.com/33522459/164789213-3e6ed9ef-43d4-4714-85a5-c98b42cbb720.png)

To train a Neural Network:
Step 1: Set up an AWS account and create an S3 bucket. Store the files in this repository in your S3 bucket.
Step 2: Pull train.sh into your desired computer using AWS CLI.
Step 3: Run train.sh. When training is completed, your trained models should be in the "models/" folder of your bucket.
This code automatically model uses 
  

![arcfaceloss](https://user-images.githubusercontent.com/33522459/164778056-9dcd3f6d-03f6-4374-a17a-ac8687b81c4a.png)
