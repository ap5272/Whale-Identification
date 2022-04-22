#!/bin/sh


#aws s3 cp s3://comp123456/train.sh ~
#sudo yum install dos2unix
#dos2unix train.sh
#chmod +x train.sh
#screen sh train.sh 

# ctrl+a, d
#screen -ls
#screen -r 


sudo yum update 
pip3 install python-box timm pytorch-lightning==1.4.0 tqdm ttach pandas sklearn torchmetrics==0.5.0 kaggle

mkdir ~/.kaggle

aws s3 cp s3://comp123456/kaggle.json ~/.kaggle/kaggle.json
chmod 600 ~/.kaggle/kaggle.json

kaggle datasets download -d bdsaglam/happy-whale-512

unzip ~/happy-whale-512.zip 

rm happy-whale-512.zip

aws s3 cp s3://comp123456/whale_train.py ~




chmod +x whale_train.py

python3 whale_train.py &

aws s3 cp ~/swin_small_patch4_window7_224/default/ s3://comp123456/models/ --recursive

sudo shutdown -h now














