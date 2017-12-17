models='InceptionV3 InceptionResNetV2'
for model in $models
do
python train.py \
  --model $model \
  --lr 5e-04 \
  --optimizer "SGD"
done
