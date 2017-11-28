cd ..

mkdir dog_breed_datasets
cd dog_breed_datasets
wget --load-cookies cookies.txt \
        https://www.kaggle.com/c/dog-breed-identification/download/labels.csv.zip \
        https://www.kaggle.com/c/dog-breed-identification/download/sample_submission.csv.zip \
        https://www.kaggle.com/c/dog-breed-identification/download/train.zip \
        https://www.kaggle.com/c/dog-breed-identification/download/test.zip

unzip '*.zip'
rm *.zip

cd ../Dog-Breed-Identification
