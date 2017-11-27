cd ..

mkdir dog_breed_datasets
cd dog_breed_datasets
wget 'https://www.kaggle.com/c/dog-breed-identification/download/labels.csv.zip'
wget 'https://www.kaggle.com/c/dog-breed-identification/download/sample_submission.csv.zip'
wget 'https://www.kaggle.com/c/dog-breed-identification/download/train.zip'
wget 'https://www.kaggle.com/c/dog-breed-identification/download/test.zip'

unzip *.zip
rm *.zip

cd ../Dog-Breed-Identification

python dog_breed.py
