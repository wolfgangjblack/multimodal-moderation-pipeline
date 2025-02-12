import os
import re
import math
import torch
import requests
import numpy as np
import pandas as pd
from torch import nn
import seaborn as sns
from tqdm import tqdm
from PIL import Image
from io import BytesIO
import concurrent.futures
from ultralytics import YOLO
from typing import Dict, List
from datetime import datetime
from collections import Counter
import matplotlib.pyplot as plt
from torchvision import models, transforms
from sklearn.metrics import confusion_matrix, classification_report
from transformers import AutoModelForSequenceClassification, AutoTokenizer, ViTImageProcessor, ViTForImageClassification

def clean_text(text:str) -> str:
        """
        This method cleans prompt data, removing extraneous punctuation meant to denote blending, loras, or models without removing names or tags. 
        We also get rid of extraneous spaces or line breaks to reduce tokens and maintain as much semantic logic as possible
        """
        text = str(text)
        # Remove additional characters: ( ) : < > [ ]
        cleaned_text = re.sub(r'[():<>[\]]', ' ', text)
        cleaned_text = cleaned_text.replace('\n', ' ')
        # Replace multiple spaces with a single space
        cleaned_text = re.sub(r'\s+', ' ', cleaned_text)
        cleaned_text = re.sub(r'\s*,\s*', ', ', cleaned_text)

        return cleaned_text.strip()

def load_mixture_local(models_dir: str) -> Dict[str, nn.Module]:
    mixtureDict = {}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for mod in [i for i in os.listdir(models_dir) if os.path.isdir(os.path.join(models_dir,i))]:
        if 'md' in mod:
            continue

        if 'resnet' in mod.lower():
            ##First load Resnet Models
            path = os.path.join(os.path.join(models_dir,mod), 'best_model_params.pt')
            if '18' in mod:
                ##Check if resnet18 or 50
                tmp_mod = models.resnet18(weights = 'IMAGENET1K_V1')
            else:
                tmp_mod = models.resnet50(weights = 'IMAGENET1K_V1')
            num_ftrs = tmp_mod.fc.in_features
            tmp_mod.fc = nn.Linear(num_ftrs, 5)
            tmp_mod.load_state_dict(torch.load(path, map_location = device))
            tmp_mod.eval()
            tmp_mod.to(device)
            mixtureDict[mod] = tmp_mod

        elif 'yolo' in mod.lower():
            ##Load YOLO models
            path = os.path.join(os.path.join(models_dir,mod), 'best_model_params.pt')
            tmp_mod = YOLO(path, task = 'classify')
            tmp_mod.to(device)
            mixtureDict[mod] = tmp_mod
        
        elif 'tag' in mod.lower():
            ##Check if Bert/roberta model has tags or not
            mixtureDict[mod] = {'model': AutoModelForSequenceClassification.from_pretrained(os.path.join(models_dir,mod)),
                                'tokenizer': AutoTokenizer.from_pretrained(os.path.join(models_dir,mod))}
        elif 'prompt' in mod.lower():
            ##Load bert/roberta if no tag
            mixtureDict[mod] = {'model': AutoModelForSequenceClassification.from_pretrained(os.path.join(models_dir,mod)),
                                'tokenizer': AutoTokenizer.from_pretrained(os.path.join(models_dir,mod))}
        elif 'vit' in mod.lower():
            ##Load Vision Transformer
            mixtureDict[mod] = {'model': ViTForImageClassification.from_pretrained(os.path.join(models_dir,mod)),
                                'processor': ViTImageProcessor.from_pretrained(os.path.join(models_dir,mod))}

        else:
            print(f"Model {mod} not recognized")
            continue
    return mixtureDict    

def mixture_inference(mixtureDict: Dict[str, nn.Module], 
                      image: Image, prompt: str, tags: str, 
                      class_names: List[str] = ['PG', 'PG13', 'R', 'X', 'XXX']
                      ) -> Dict[str, str]:
    preds = {}

    transform = transforms.Compose([transforms.Resize((256, 256)),  
                    transforms.CenterCrop((224, 224)), 
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])])
    
    softmax = nn.Softmax(dim=1)
    models = list(mixtureDict.keys())    

    for mod in models:
        if 'resnet' in mod.lower():
            ##Do inference for resnet models
            input = transform(image).unsqueeze(0)
            input = input.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

            with torch.no_grad():  # No need to compute gradients during inference
                output = mixtureDict[mod](input)
                _, prediction = torch.max(output, 1)
                predicted_class = class_names[prediction.item()]

        elif 'yolo' in mod.lower():
            ##Do inference with YOLO models
            results = mixtureDict[mod].predict(image,
                    max_det = 1)[0]
            if 'cls' in mod.lower():
                predicted_class = results.names[results.probs.top1]
            else:
            
                predicted_class = results.names[results.boxes.cls.item()]
            
        elif 'vit' in mod.lower():
            ## Do inference for vision transformer models
            processed_input = mixtureDict[mod]['processor'](images=image, return_tensors="pt")

            with torch.no_grad():
                outputs = mixtureDict[mod]['model'](processed_input['pixel_values']).logits
                prediction = torch.argmax(outputs, dim =1)
                predicted_class = class_names[prediction.item()]
            
        else:
            ## Do inference for bert/roberta models
            if 'tag' in mod.lower():
                if prompt is None or pd.isna(prompt):
                    input = tags
                elif tags:
                    # input = prompt + ', ' + tags #Note: this was muted because I started saving tags onto the end of the prompts in data wrangling
                    input = tags
                else:
                    input = prompt
            else:
                 input = prompt
            cleaned_text = clean_text(input)
            tokens = mixtureDict[mod]['tokenizer'](cleaned_text, max_length = 512, truncation = True, padding = 'max_length', return_tensors = 'pt')

            with torch.no_grad():

                for key in tokens:
                    tokens[key]  = tokens[key]

                outputs = mixtureDict[mod]['model'](**tokens)
                logits = outputs.logits
                probs = softmax(logits)
                _, prediction = torch.max(probs,1)
            
            prediction = prediction.item()
        
            predicted_class = mixtureDict[mod]['model'].config.id2label[prediction]

        preds[mod] = predicted_class

    return preds

def inference_voting(mylist: List[int]) -> int:
    """
    A function used to determine the most common pred among the N-odd models
    in cases of tie, returns the most conservative answer
    """

    return sorted(Counter(mylist).most_common(), key = lambda x: (x[1], x[0]))[-1][0]

def multimodal_batch_predict(model,
            dataloader,
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
    
    model.to(device)
    model.eval()

    results = {"true_labels": [],
    "predictions": [],
    "file_paths": []
    }

    with torch.no_grad():
        for batch in tqdm(dataloader, desc= "Predicting"):
              # Extract inputs and labels from batch
                resnet18 = batch['resnet18'].to(device)
                resnet50 = batch['resnet50'].to(device)
                vit = batch['vit'].squeeze(1).to(device)
                bert = {key: val.to(device) for key, val in batch['bert'].items()}
                roberta = {key: val.to(device) for key, val in batch['roberta'].items()}
                labels = batch['label'].to(device)

                # Forward pass
                outputs = model(
                    resnet18=resnet18,
                    resnet50=resnet50,
                    vit=vit,
                    bert=bert,
                    roberta=roberta,
                    labels=labels
                )

                _, preds = torch.max(outputs['logits'], dim = 1)
                results['predictions'].extend(preds.cpu().numpy())
                results['true_labels'].extend(labels.cpu().numpy())
                results['file_paths'].extend(batch['filepath'])

    return results

def get_image(url):
    try:
        response = requests.get(url)
        if response.status_code == 200:
            image = Image.open(BytesIO(response.content))
            if image.mode == "RGBA":
                image = image.convert("RGB")
            return image
    except Exception as e:
        print(f"Error processing image from {url}: {e}")
        return None    
    
def process_row(row: pd.Series, mixtureDict: Dict[str, nn.Module]) -> Dict[str, str]:

    url = row['new_url']
    prompt = row['prompt']
    id = row['id']
    label = row['label']
    tags = row['tags']

    try: 
        image = get_image(url)
    except Exception as e:
        print(f"got error in download: {e} on image {id}")
        return None
    
    try:
        preds = mixture_inference(mixtureDict, image, prompt, tags)
        pred = inference_voting(list(preds.values()))
    except Exception as e:
        print(f"got error in download: {e} on image {id}")
        return None
    print(f"success {id}")
    return {'new_url': url,
            "prompt": prompt,
            "prediction": pred,
            "label": label,
            'model_preds': preds,
            'tags': tags}

def build_eval_dataframe_with_concurrency(eval_data: pd.DataFrame, mixtureDict: Dict[str, nn.Module],
                                          columns: List[str] = ['new_url', "prompt", "tags", "prediction", 'label', 'model_preds'], 
                                          max_workers: int =8) -> pd.DataFrame:
    """
    Function to build a dataframe with the results of the evaluation of the model as well as the main features for the models
    in the mixtureDict
    
    Warnings: Users beware the fields in the eval_data need to match the fields in the process_row function
    and this output
    """
    with concurrent.futures.ThreadPoolExecutor(max_workers) as executor:
        futures =[]
        for _, row in eval_data.iterrows():
            futures.append(executor.submit(process_row, row, mixtureDict))
            
        results = []
        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            if result:
                results.append(result)

    evalDict = {col: [] for col in columns}
    
    for i in results:
        for k,v in i.items():
            evalDict[k].append(v)

    return pd.DataFrame(evalDict)
    
def get_mixture_predictions_from_list(dataframe: pd.DataFrame, model_list: List[str]):
    """
    Use this to generate new mixtures from individual model scores in a dataframe
    """
    scores = []
    for i in model_list:
        scores.append(list(dataframe[i]))

    baseMixtureScores = [inference_voting(j) for j in [list(i) for i in list(np.transpose(scores))]]
    
    return baseMixtureScores

def plot_confusion_matrix(true_labels, predicted_labels, classes) -> None:
    # Compute confusion matrix
    cm = confusion_matrix(true_labels, predicted_labels)
    
    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')
    plt.title('Confusion Matrix')
    plt.show()


def plot_confusion_matrix_norm(true_labels, predicted_labels, classes) -> None:
    cm = confusion_matrix(true_labels, predicted_labels, normalize='true')
    
    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='.2%', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')
    plt.title('Confusion Matrix')
    plt.show()


def plot_training_history(lossAccDict: dict) -> None:
    """
    After training plot training history for pytorch ResNet finetuning
    - Note: If earlystopping is activated, the validation list will be sort
        one value, this fixes that by copying the last recored validation
        loss and accuracy and appending it at the end
    """
    train_loss_history = lossAccDict['train']['loss']
    train_acc_history = lossAccDict['train']['acc']
    val_loss_history = lossAccDict['val']['loss']
    val_acc_history = lossAccDict['val']['acc']

    if len(val_acc_history) < len(train_acc_history):
        val_loss_history.append(val_loss_history[-1])
        val_acc_history.append(val_acc_history[-1])
        print("early stopping was activated so the validation loss and accuracy were extended by 1")

    try:
        train_acc_history = [i.to('cpu') for i in train_acc_history]
    except TypeError:
        pass
    try:
        val_acc_history = [i.to('cpu') for i in val_acc_history]
    except TypeError:
        pass

    epochs = range(1, len(train_loss_history) + 1)

    plt.figure(figsize=(10, 5))

    # Plot loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_loss_history, label='Train')
    plt.plot(epochs, val_loss_history, label='Validation')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    # Plot accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_acc_history, label='Train')
    plt.plot(epochs, val_acc_history, label='Validation')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.show()

def plot_all_class_disagreements(dataframe: pd.DataFrame, label_col: str = 'label', 
        pred_col: str = 'prediction', true_label: str = 'XXX', pred_label: str = 'PG',
        num_cols=3, img_width=4, img_height=4, output_dir: str = '../results/',
        ) -> None:
    
    ##Check outputdir
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    ## Get date for filename
    today = datetime.today().strftime("%m_%d_%y_")

    ##create output_file
    output_file = f'{today}model_{pred_col}_{true_label}_mislabeled_as_{pred_label}.png'
    
    # Filter the DataFrame for the disagreeing images
    subset = dataframe[
        (dataframe[label_col] == true_label) &
        (dataframe[pred_col] == pred_label)]
    
    # Calculate the number of images and rows needed
    num_imgs = len(subset)
    num_rows = math.ceil(num_imgs / num_cols)
    
    # Create subplots
    fig, axs = plt.subplots(num_rows, num_cols, 
                figsize=(num_cols * img_width, num_rows * img_height))
    fig.suptitle(f"Model type: {pred_col} -> {true_label} imgs mislabeled as {pred_label}")
    plt.subplots_adjust(top=0.9, hspace=0.5, wspace=0.5)  # Adjust spacing as needed

    # Flatten axs for easier indexing
    axs = axs.flatten()

    # Plot each image in the subset
    for i, row in enumerate(subset.itertuples()):
        img = get_image(row.download_url)
        axs[i].imshow(img)
        axs[i].axis('off')

    # Turn off any unused subplots
    for j in range(num_imgs, len(axs)):
        axs[j].axis('off')

    # Show the figure
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, output_file))
    plt.show()

def save_classification_reports(dataframe: pd.DataFrame, mixtures: List[str] = ['prediction', 'AprilMixtureModel', 'ViTLanMixture', 'ViTBertMixture'],
                                true_label_col: str = 'label', output_dir: str = '../results/',
                                output_file: str = 'classification_report.txt') -> None:
     
    ##Check outputdir
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    ## Get date for filename
    today = datetime.today().strftime("%m_%d_%y_")

    class_report = ""
    for model in mixtures:
        class_report += model + ' Classification report \n'
        class_report += classification_report(dataframe[true_label_col], dataframe[model])
        class_report += '*******\n\n'
    
    print(class_report)
    with open(os.path.join(output_dir, today + output_file), 'w') as f:
        f.write(class_report)

def save_markdown_tables(dataframe: pd.DataFrame, mixtures: List[str] = ['prediction', 'AprilMixtureModel', 'ViTLanMixture', 'ViTBertMixture'],
                        true_label_col: str = 'label', classes: List[str] = ['PG', 'PG13', "R", "X", "XXX"],
                        output_dir: str = '../results/', output_file: str = 'markdown_tables.md') -> None:
         
    ##Check outputdir
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    ## Get date for filename
    today = datetime.today().strftime("%m_%d_%y_")

    longstr = ''
    for mod in mixtures:
        longstr += f'#### {mod}\n'
        longstr += '|Class|Name|TP|FP|TN|FN|Accuracy|Precision|Recall|F1|\n'
        longstr += '|---|---|---|---|---|---|---|---|---|---|\n'
        cm = confusion_matrix(dataframe[true_label_col], dataframe[mod])
        for i, cls in enumerate(classes):
            tp = cm[i,i]
            fp = sum(cm[:,i])-cm[i,i]
            fn = sum(cm[i,:])-cm[i,i]
            tn = sum(sum(cm))-tp-fp-fn
            acc = np.round(100*(tp+tn)/(tp+tn+fp+fn),2)
            pre = np.round(tp/(tp+fp),2)
            rec = np.round((tp)/(tp+fn),2)
            f1 = np.round(2*tp/(2*tp+fp+fn),2)
            longstr += f'| {i} | {cls} | {tp} | {fp} | {tn} | {fn} | {acc} | {pre} | {rec} | {f1} |\n'
        longstr += '\n********\n\n'
    print(longstr)
    with open(os.path.join(output_dir, today + output_file), 'w') as f:
        f.write(longstr)

def plot_confusion_matrix_subplots(dataframe: pd.DataFrame, mixtures: List[str] = ['prediction', 'AprilMixtureModel', 'ViTLanMixture', 'ViTBertMixture'],
        ncols: int = 2, true_label_col: str= 'label', classes: List[str] = ['PG', "PG13", "R", "X", "XXX"], output_dir: str = '../results/',
        output_file: str = 'mixture_subplots_confusion_matrix.png', norm = False):
    
    ##Check outputdir
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    ## Get date for filename
    today = datetime.today().strftime("%m_%d_%y_")

    ncols = 2
    nrows = math.ceil(len(mixtures) / ncols)

    fig, axs = plt.subplots(nrows, ncols, figsize=(15, 15))

    axs = axs.flatten()

    if norm:
        output_file = 'norm_'+output_file

    for i, col in enumerate(mixtures):
        if norm:
            cm = confusion_matrix(dataframe[true_label_col], dataframe[col], normalize='true')

        else:
            cm = confusion_matrix(dataframe[true_label_col], dataframe[col])

        axs[i].set_title(col)
        sns.heatmap(cm, cmap='Blues', xticklabels =classes,
                    yticklabels =classes, ax=axs[i], annot=True, fmt='d' if not norm else '.2%')
        axs[i].set_xlabel('Predicted')
        axs[i].set_ylabel('True')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, today+output_file))
    plt.show()

def save_all_class_disagreements_plot(dataframe: pd.DataFrame, label_col: str = 'label', 
        pred_col: str = 'prediction', true_label: str = 'XXX', pred_label: str = 'PG',
        num_cols=3, img_width=4, img_height=4, output_dir: str = '../results/',
        ) -> None:
    
    ##Check outputdir
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    ## Get date for filename
    today = datetime.today().strftime("%m_%d_%y_")

    ##create output_file
    output_file = f'{today}model_{pred_col}_{true_label}_mislabeled_as_{pred_label}.png'
    
    # Filter the DataFrame for the disagreeing images
    subset = dataframe[
        (dataframe[label_col] == true_label) &
        (dataframe[pred_col] == pred_label)]
    
    # Calculate the number of images and rows needed
    num_imgs = len(subset)
    num_rows = math.ceil(num_imgs / num_cols)
    
    # Create subplots
    try:
        fig, axs = plt.subplots(num_rows, num_cols, 
                figsize=(num_cols * img_width, num_rows * img_height))
        fig.suptitle(f"Model type: {pred_col} -> {true_label} imgs mislabeled as {pred_label}")
        plt.subplots_adjust(top=0.9, hspace=0.5, wspace=0.5)  # Adjust spacing as needed

        # Flatten axs for easier indexing
        axs = axs.flatten()

        # Plot each image in the subset
        for i, row in enumerate(subset.itertuples()):
            img = get_image(row.download_url)
            axs[i].imshow(img)
            axs[i].axis('off')

        # Turn off any unused subplots
        for j in range(num_imgs, len(axs)):
            axs[j].axis('off')

        # Show the figure
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, output_file))
        plt.show()
    except Exception as e:
        print(f"got error in setting up plot: {e}")
        return None
    