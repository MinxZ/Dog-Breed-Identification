rm *.h5
rm *.csv

python train.py \
  --model 'Xception' \
  --lr 1e-04 \
  --optimizer "Nadam" 
  
python train.py \
  --model 'Xception' \
  --lr 5e-05 \
  --optimizer "Nadam" 
  
python train.py \
  --model 'Xception' \
  --lr 5e-04 \
  --optimizer "SGD" 
  
python train.py \
  --model 'Xception' \
  --lr 1e-04 \
  --optimizer "SGD" \
  --patience 2

python train.py \
  --model 'Xception'
  --lr 5e-05 \
  --optimizer "SGD" \
  --patience 2

python train.py \
  --model 'InceptionV3' \
  --lr 1e-04 \
  --optimizer "Nadam" 
  
python train.py \
  --model 'InceptionV3' \
  --lr 5e-05 \
  --optimizer "Nadam" 
  
python train.py \
  --model 'InceptionV3' \
  --lr 5e-04 \
  --optimizer "SGD" 
  
python train.py \
  --model 'InceptionV3' \
  --lr 1e-04 \
  --optimizer "SGD" \
  --patience 2

python train.py \
  --model 'InceptionV3'
  --lr 5e-05 \
  --optimizer "SGD" \
  --patience 2

python train.py \
  --model 'InceptionResNetV2' \
  --lr 1e-04 \
  --optimizer "Nadam" 
  
python train.py \
  --model 'InceptionResNetV2' \
  --lr 5e-05 \
  --optimizer "Nadam" 
  
python train.py \
  --model 'InceptionResNetV2' \
  --lr 5e-04 \
  --optimizer "SGD" 
  
python train.py \
  --model 'InceptionResNetV2' \
  --lr 1e-04 \
  --optimizer "SGD" \
  --patience 2

python train.py \
  --model 'InceptionResNetV2'
  --lr 5e-05 \
  --optimizer "SGD" \
    --patience 2


