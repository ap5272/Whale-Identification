#!/bin/sh

sudo yum update 

#aws s3 cp s3://comp123456/test.py ~
#sudo yum install dos2unix
#dos2unix main.sh
#chmod +x main.sh

#screen sh main.sh 
# ctrl+a, d
#screen -ls
#screen -r 12345

#nohup sh main.sh &> out.txt &
#cat out.txt

pip3 install pandas
pip3 install matplotlib
pip3 install sklearn
pip3 install xgboost

pip3 install scipy

aws s3 cp s3://comp123456/quant_train.py ~

aws s3 cp s3://comp123456/train.pkl ~/train.pkl

mkdir models

chmod +x quant_train.py

python3 quant_train.py

aws s3 cp ~/models/ s3://comp123456/models/ --recursive

sudo shutdown -h now














