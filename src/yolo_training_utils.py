import os
from sklearn.model_selection import train_test_split

def split_data_yolo(data_dir, classes = None, train_size=0.8, val_size=0.1, test_size=0.1):
    if classes == None:
        classes = os.listdir(data_dir)
    train_list, val_list, test_list = [], [], []
    
    for cls in classes:
        cls_path = os.path.join(data_dir, cls)
        images = os.listdir(cls_path)
        
        # Split data
        train, test = train_test_split(images, test_size=(1-train_size))
        val, test = train_test_split(test, test_size=test_size/(test_size + val_size))
        
        # Add full paths
        train_list.extend([os.path.join(cls_path, img) for img in train])
        val_list.extend([os.path.join(cls_path, img) for img in val])
        test_list.extend([os.path.join(cls_path, img) for img in test])
    
    return train_list, val_list, test_list