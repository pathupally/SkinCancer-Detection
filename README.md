chmod +x data_pipeline/data_request.sh

pip3 install -r requirements.txt

python3 src/data_org.py

python3 train.py