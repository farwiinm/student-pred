import os
import sys
import dill

from src.exception import CustomException

def save_object(file_path, obj):
    """
    This function saves the object to a file using dill.
    """
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'wb') as file_obj:
            dill.dump(obj, file_obj)
        print(f"Object saved successfully at {file_path}")
    except Exception as e:
        raise CustomException(e, sys)