rm Xception.h5 \
  InceptionV3.h5 \
  InceptionResNetV2

rm *.csv

python train.py \
  --model "Xception" \
  --lr 1e-03 \
  --optimizer "Adam" \
  --patience 5

models='Xception InceptionV3 InceptionResNetV2'
for model in $models
do
python train.py \
  --model $model \
  --lr 1e-04 \
  --optimizer "Adam"

python train.py \
  --model $model
  --lr 1e-04 \
  --optimizer "SGD"
done
