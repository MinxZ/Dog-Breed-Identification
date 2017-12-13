rm Xception.h5 \
  InceptionV3.h5 \
  InceptionResNetV2

rm *.csv

<<<<<<< HEAD
python train.py \
  --model "Xception" \
  --lr 1e-03 \
  --optimizer "Adam" \
  --patience 5
=======
# python train.py \
#   --model "Xception" \
#   --lr 1e-04 \
#   --optimizer "SGD" \
#   --patience 5
>>>>>>> f2f416d56c101acd584378a86922ab2e62a7e552

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
