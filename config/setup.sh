pip3 install -r requirements

mkdir ./data/raw
mkdir ./data/raw/images

isic image download -l 10000 data/raw/images/
mv data/raw/images/metadata.csv data/raw


