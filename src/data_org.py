import pandas as pd
import os
import shutil
from sklearn.model_selection import train_test_split
import logging
from datetime import datetime
from utils import setup_logger

logger = setup_logger()

def load_metadata(data_dir='data') -> pd.DataFrame:
    logging.info("Loading metadata")
    df = pd.read_csv(f"{data_dir}/raw/metadata.csv")
    needed_columns = ['isic_id', 'age_approx', 'diagnosis_1']
    meta_data = df[needed_columns].dropna()
    meta_data = meta_data[meta_data['diagnosis_1'] != 'Indeterminate']
    df = df.reset_index(drop=True)
    return meta_data

def sort_images(df, data_dir='data'):
    logging.info("Starting image sorting")
    
    train_df, test_df = train_test_split(
        df,
        test_size=0.2,
        stratify=df['diagnosis_1'],
        random_state=42
    )
    

    val_df, test_df = train_test_split(
        test_df,
        test_size=0.5,
        stratify=test_df['diagnosis_1'],
        random_state=42
    )
    
    source_folder = os.path.join(data_dir, 'raw/images')
    base_output = os.path.join(data_dir, 'processed')
    split_map = [('train', train_df), ('val', val_df), ('test', test_df)]
    
    for split, split_df in split_map:
        logging.info(f"Processing {split} split with {len(split_df)} images")
        for _, row in split_df.iterrows():
            img_name = row['isic_id'] + '.jpg'
            label = row['diagnosis_1']
            src_path = os.path.join(source_folder, img_name)
            dest_dir = os.path.join(base_output, split, label)
            os.makedirs(dest_dir, exist_ok=True)
            dest_path = os.path.join(dest_dir, img_name)
            if os.path.exists(src_path):
                shutil.copy2(src_path, dest_path)
            else:
                print(f"Image {src_path} not found!")
                
    return train_df, val_df, test_df


df = load_metadata()
sort_images(df)