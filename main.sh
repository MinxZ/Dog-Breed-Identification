rm Xception.h5 \
  InceptionV3.h5 \
  InceptionResNetV2
  
rm *.csv

models='Xception InceptionV3 InceptionResNetV2'
for model in $models
do
python train.py \
  --model $model \
  --lr 1e-04 \
  --optimizer "Nadam"

python train.py \
  --model $model \
  --lr 5e-05 \
  --optimizer "Nadam"

python train.py \
  --model $model \
  --lr 5e-04 \
  --optimizer "SGD"

python train.py \
  --model $model \
  --lr 1e-04 \
  --optimizer "SGD" \
  --patience 2

python train.py \
  --model $model
  --lr 5e-05 \
  --optimizer "SGD" \
  --patience 2
done

