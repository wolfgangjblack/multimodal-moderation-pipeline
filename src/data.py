import os
import torch
import requests
from PIL import Image
from io import BytesIO
import concurrent.futures
from torch.utils.data import Dataset

class CustomImageTextDataset(Dataset):
    def __init__(self, data_dir, target_transform=None):
        self.data_dir = data_dir        
        self.files = self.get_files()
        self.labels = self.get_files(True)
        self.target_transform = target_transform
        
    def get_files(self, labels = False):
        files = []
        for i in os.listdir(self.data_dir):
            if labels:
                imgs = [i for j in os.listdir(os.path.join(self.data_dir,i)) if j.endswith(".jpg")]
            else:
                imgs = [os.path.join(self.data_dir, i, j.split('.')[0]) for j in os.listdir(os.path.join(self.data_dir,i)) if j.endswith(".jpg")]
            files.extend(imgs)
        return files
    
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        img_path = self.files[idx] + '.jpg'
        txt_path = self.files[idx] + '.txt'
        tag_path = self.files[idx] + '_tags.txt'
        # image = Image.open(img_path)
        try:
            text = open(txt_path, 'r').read()
        except:
            text = ''
        try:
            tags = open(tag_path, 'r').read()
        except:
            tags = None
        label = self.labels[idx]
        if self.target_transform:
            label = self.target_transform[label]
            if isinstance(label, int):
                label = torch.tensor(label)
        return img_path, text, tags, label
    
    def __repr__(self):
        string = f"CustomImageTextDataset(num_samples={len(self.files)}, data_dir='{self.data_dir}')\n"
        return string + f"Unique Labels: {list(set(self.labels))} for phase: {os.path.basename(self.data_dir)}"
    

def get_data_for_training(data_item, output_dir):
    url = data_item['url']
    prompt = data_item.get('prompt', None)
    prompt_and_tags = data_item.get('prompt_and_tags', None)
    folder = data_item['label']
    
    try:
        response = requests.get(url)
        if response.status_code == 200:
            image = Image.open(BytesIO(response.content))
            if image.mode == "RGBA":
                image = image.convert("RGB")
            
            image_name = url.split('/')[-1]
            print(image_name)
            path = os.path.join(output_dir, folder)
            os.makedirs(path, exist_ok=True)
            image_path = os.path.join(path, image_name)
            prompt_path = os.path.join(path, f"{image_name.split('.')[0]}.txt")
            prompt_and_tags_path = os.path.join(path, f"{image_name.split('.')[0]}_tags.txt")
            
            image.save(image_path)

            #save prompt
            if prompt is not None:
                with open(prompt_path, 'w') as f:
                    f.write(prompt)

            #save prompt and tags
            if prompt_and_tags is not None:
                with open(prompt_and_tags_path, 'w') as f:
                    f.write(prompt_and_tags)

    except Exception as e:
        print(f"Error processing image from {url}: {e}")

def download_images(imageUrlDict, output_dir, max_workers=8):
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for index, data_item in imageUrlDict.items():

            futures.append(executor.submit(get_data_for_training, data_item, output_dir))
        
        for future in concurrent.futures.as_completed(futures):
            future.result()

