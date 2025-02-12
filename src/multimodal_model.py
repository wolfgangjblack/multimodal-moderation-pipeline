import re
import os
import torch
from torch import nn
from PIL import Image
from ultralytics import YOLO
from torchvision import models, transforms
from transformers import (PretrainedConfig, ViTImageProcessor, ViTForImageClassification,
                          AutoTokenizer, AutoModelForSequenceClassification, PreTrainedModel)
from safetensors.torch import save_file, load_file

class MultimodalConfig(PretrainedConfig):
    model_type = "multimodal"

    def __init__(self,
                 models=None,
                 **kwargs):
        super().__init__(**kwargs)
        self.models = models if models is not None else {}
        self.image_models = [i for i in self.models.keys() if 'prompt' not in i]

        self.resnet_model_paths = {i:v for i,v in self.models.items() if 'resnet' in i.lower()}
        self.vit_model_paths = {i:v for i,v in self.models.items() if 'vit' in i}
        self.yolo_model_paths = {i:v for i,v in self.models.items() if 'yolo' in i}

        self.text_models = [i for i in self.models.keys() if 'prompt' in i]
        self.nlp_transformers_model_paths = {i:v for i,v in self.models.items() if 'prompt' in i}
        
        self.class_names = ["PG", "PG13", "R", "X", "XXX"]
        self.label2id = {label: i for i, label in enumerate(self.class_names)}
        self.id2label = {i: label for label, i in self.label2id.items()}
        
    def update_model_paths(self, models):
        self.models = models
        self.resnet_model_paths = {i:v for i,v in self.models.items() if 'resnet' in i.lower()}
        self.vit_model_paths = {i:v for i,v in self.models.items() if 'vit' in i}
        self.yolo_model_paths = {i:v for i,v in self.models.items() if 'yolo' in i}
        self.text_models = [i for i in self.models.keys() if 'prompt' in i]
        self.nlp_transformers_model_paths = {i:v for i,v in self.models.items() if 'prompt' in i}
        
        return None
    
class MultimodalProcessor:
    def __init__(self, models = None):
        self.models = models if models is not None else {}
        self.image_models = [i for i in self.models.keys() if 'prompt' not in i]
        self.text_models = [i for i in self.models.keys() if 'prompt' in i]
        self.image_processors = self.load_image_processors()
        self.text_processors = self.load_text_processors()
    
    @staticmethod
    def clean_text(text: str) -> str:
        text = str(text)
        cleaned_text = re.sub(r"[():<>[\]]", " ", text)
        cleaned_text = cleaned_text.replace("\n", " ")
        cleaned_text = re.sub(r"\s+", " ", cleaned_text)
        cleaned_text = re.sub(r"\s*,\s*", ", ", cleaned_text)
        return cleaned_text.strip()         
    
    def load_image_processors(self):
        """Use either transforms native to the model architecture or the vit image processor
        NOTE: if you change the transforms an image undergoes during training, you should change them here as well
        """
        image_processors = {}
        for model_name in self.image_models:
            if 'vit' in model_name.lower():
                try:
                    processor = ViTImageProcessor.from_pretrained(self.models[model_name])
                    image_processors[model_name] = processor
                except Exception as e:
                    print(f"Error loading {model_name}: {e}")
            elif 'resnet' in model_name.lower() and 'resnet' not in image_processors.keys():
                try:
                    processor = transforms.Compose([
                        transforms.Resize(256),
                        transforms.CenterCrop(224),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                    ])
                    image_processors['resnet'] = processor
                except Exception as e:
                    print(f"Error loading {model_name}: {e}")
            elif 'resnet' in model_name.lower() and 'resnet' in image_processors.keys():
                continue
            elif 'yolo' and 'cls' in model_name.lower():
                image_processors[model_name] = YOLO(os.path.join(self.models[model_name], 'best_model_params.pt')).transforms
            else:
                #We expect DET Yolos to be None, as they just
                #take in the PIL.Image and return the output
                image_processors[model_name] = None
        return image_processors    

    def load_text_processors(self):
        text_processors = {}
        for model_name in self.text_models:
            try:
                tokenizer = AutoTokenizer.from_pretrained(self.models[model_name])
                text_processors[model_name] = tokenizer
            except Exception as e:
                print(f"Error loading {model_name}: {e}")
        return text_processors
    
    def __call__(self, img, text, tags, label = None):
        inputs = {}
        
        for model in self.image_models:
            if 'vit' in model.lower():
                inputs[model] = self.image_processors[model](img,
                        return_tensors="pt")['pixel_values']
        
            elif 'resnet' in model.lower():
                inputs[model] = self.image_processors['resnet'](img).unsqueeze(0)
        
            elif 'yolo' and 'cls' in model.lower():
                inputs[model] = self.image_processors[model](img)
            else:
                inputs[model] = img_path
        
        for model in self.text_models:
            if 'tag' in model.lower() and tags:
                text_in = ' '.join([text, tags])
            else:
                text_in = text
            
            text_in = self.clean_text(text_in)
            inputs[model] = self.text_processors[model](text_in, return_tensors="pt", truncation=True, padding='max_length')
        
        inputs['labels'] = label 
        return inputs

class MultimodalModel(PreTrainedModel):
    config_class = MultimodalConfig

    def __init__(self, config: MultimodalConfig,
                 device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
                 ):
        super().__init__(config)
        self.num_classes = 5  
        self.features = {}
        
        # Initialize the models and set requires_grad = False to freeze them
        self.resnet_models = nn.ModuleDict()
        if config.resnet_model_paths:
            for model_id, model_path in config.resnet_model_paths.items():
                ##Load model architecture
                model = models.resnet50() if "resnet50" in model_id.lower() else models.resnet18()
                prev_fc = model.fc
                model.fc = nn.Linear(model.fc.in_features, self.num_classes)
                ##Load weights
                model.load_state_dict(torch.load(f"{model_path}/best_model_params.pt", map_location='cpu'))
                model.fc = prev_fc
                ##set output to identity to return layer before prediction
                model.fc = nn.Identity()
                ##freeze model
                for param in model.parameters(): 
                    param.requires_grad = False
                model.to(device)
                ##add model to model dict
                self.resnet_models[model_id] = model
                ##store number of features
                self.features[model_id] = prev_fc.in_features

        self.vit_models = nn.ModuleDict()
        if config.vit_model_paths:
            for model_id, model_path in config.vit_model_paths.items():
                model = ViTForImageClassification.from_pretrained(model_path)
                for param in model.parameters():
                    param.requires_grad = False
                model.to(device)
                self.vit_models[model_id] = model
                self.features[model_id] = model.classifier.in_features

        self.yolo_models = nn.ModuleDict()
        if config.yolo_model_paths:
            for model_id, model_path in config.yolo_model_paths.items():
                if 'cls' in model_id:
                    model = YOLO(os.path.join(model_path, 'best_model_params.pt'))
                    sequential_model = model.model.model
                    self.features[model_id] = sequential_model[-1].linear.in_features
                    
                    self.replace_yolo_last_linear_with_identity(sequential_model)
                    for param in sequential_model.parameters():
                        param.requires_grad = False
                    model.to(device)                        
                    self.yolo_models[model_id] = sequential_model

                elif 'det' in model_id:
                    model = YOLO(os.path.join(model_path, 'best_model_params.pt'))
                    model.to(device)
                    self.yolo_models[model_id] = model
                    self.features[model_id] = None
                    
        self.nlp_transformers_models = nn.ModuleDict()
        if config.nlp_transformers_model_paths:
            for model_id, model_path in config.nlp_transformers_model_paths.items():
                model = AutoModelForSequenceClassification.from_pretrained(model_path)
                for param in model.parameters():
                    param.requires_grad = False
                model.to(device)
                self.nlp_transformers_models[model_id] = model
                self.features[model_id] = self.get_classifier_in_features(model)

        # MLP layers (only this part will be fine-tuned)
        self.mlp = nn.Sequential(
            nn.Linear(sum(self.features.values()), 1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, config.num_labels)
        )

        self.loss_fn = nn.CrossEntropyLoss()

    def get_classifier_in_features(self, model):
        """
        Function to get the number of input features to the classifier layer.
        Handles different model structures such as BERT and RoBERTa.
        """
        try:
            # RoBERTa's classifier
            return model.classifier.out_proj.in_features
        except AttributeError:
            pass

        try:
            # BERT's classifier
            return model.classifier.linear.in_features
        except AttributeError:
            pass

        try:
            # DistilBERT's classifier
            return model.classifier.in_features
        except AttributeError:
            pass

        try: 
            #ViT's classifier
            return model.classifier.in_features
        except AttributeError:
            pass

        try: 
            return model.fc.in_features
        except AttributeError:
            pass

        # Add other model types as needed
        raise ValueError("Unknown model structure or unsupported model type.")
    
    def forward(self, **inputs):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.mlp = self.mlp.to(device)
        features = []        
        labels = inputs.pop('labels', None)

        for key,input in inputs.items():
            input = inputs[key]
            
            if key in self.resnet_models.keys():
                with torch.no_grad():
                    resnet_features = self.resnet_models[key](input.squeeze(1).to(device))
                features.append(resnet_features)
            
            if key in self.vit_models.keys():
                with torch.no_grad():
                    vit_features = self.vit_models[key](pixel_values=input.squeeze(1).to(device),
                                            output_hidden_states=True).hidden_states[-1][:, 0, :]
                features.append(vit_features)

            if key in self.yolo_models.keys():
                with torch.no_grad():
                    yolo_features = self.yolo_models[key](input.to(device))
                features.append(yolo_features)
            
            if key in self.nlp_transformers_models.keys():
                with torch.no_grad():
                    nlp_features = self.nlp_transformers_models[key](input['input_ids'].squeeze(1).to(device),
                    input['attention_mask'].squeeze(1).to(device),
                    output_hidden_states=True).hidden_states[-1][:, 0, :]
                features.append(nlp_features)
            
        ##Concat features
        features = torch.cat(features, dim=1)
        logits = self.mlp(features)

        loss = None
        if labels is not None:
            loss = self.loss_fn(logits.view(-1, self.config.num_labels), labels.view(-1))
    
        return {"loss": loss, "logits": logits}

    @staticmethod
    def replace_yolo_last_linear_with_identity(model):
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                parent_name, child_name = name.rsplit('.',1)
                parent = model
                for part in parent_name.split('.'):
                    parent = getattr(parent, part)
                setattr(parent, child_name, nn.Identity())
                return True
        return False
    
    def save_pretrained(self, save_directory, state_dict=None, safe_serialization=True):
        os.makedirs(save_directory, exist_ok=True)
        if state_dict is None:
            state_dict = self.state_dict()
        if safe_serialization:
            try:
                save_file(state_dict, os.path.join(save_directory, "model.safetensors"))
            except Exception as e:
                print(f"Error saving with safetensors: {e}")
                print("Falling back to torch.save")
                torch.save(state_dict, os.path.join(save_directory, "pytorch_model.bin"))
        else:
            torch.save(state_dict, os.path.join(save_directory, "pytorch_model.bin"))
        self.config.save_pretrained(save_directory)
    
    @classmethod
    def from_pretrained(cls, save_directory, config = None, **kwargs):
        
        if config is None:
            config = cls.config_class.from_pretrained(save_directory)
            
        model = cls(config)
        safe_serialization = kwargs.get("safe_serialization", True)
        
        if safe_serialization:
            try:
                state_dict = load_file(os.path.join(save_directory, "model.safetensors"))
            except Exception as e:
                print(f"Error loading with safetensors: {e}")
                print("Falling back to torch.load")
                state_dict = torch.load(os.path.join(save_directory, "pytorch_model.bin"), map_location='cpu')
        else:
            state_dict = torch.load(os.path.join(save_directory, "pytorch_model.bin"), map_location='cpu')
        
        # Check if MLP weights are in the state dict
        mlp_keys = [k for k in state_dict.keys() if k.startswith('mlp.')]
        if not mlp_keys:
            print("Warning: MLP weights not found in the loaded state dict. The MLP will be randomly initialized.")
        
        # Load the state dict
        model.load_state_dict(state_dict, strict=False)
        
        return model