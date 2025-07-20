import os
import logging
from datetime import datetime
import inspect

def get_curr_filename(strip_extension=True, skip=2):
    frame = inspect.stack()[skip]
    filepath = frame.filename
    filename = os.path.basename(filepath)
    return os.path.splitext(filename)[0] if strip_extension else filename
    

def setup_logger():
    file_dir = f"./logs/{get_curr_filename()}"
    os.makedirs(file_dir, exist_ok=True)
    log_filename = datetime.now().strftime("%Y-%m-%d_%H-%M-%S.log")
    log_path = os.path.join(file_dir, log_filename)

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] [%(filename)s] %(message)s',
        handlers=[
            logging.FileHandler(log_path),
            logging.StreamHandler()
        ]
    )
