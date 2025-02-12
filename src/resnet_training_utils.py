import os
import time
import torch
import random
from torch import nn
from typing import Dict, List, Tuple
from tempfile import TemporaryDirectory
from torch.nn.modules.loss import _Loss
from torchvision import transforms
from torch.optim.optimizer import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.model_selection import StratifiedShuffleSplit

def training_resnet_model_loop(dataloaders: DataLoader, model: nn.Module,
                criterion: _Loss, optimizer: Optimizer, scheduler: _LRScheduler, 
                patience: int = 5, num_epochs: int = 25, weight_decay: float = 0.0001
                ) -> Tuple[nn.Module,Dict[str, List]]:
    """Training loop with regularization and early stopping - for best results increase patience to something like 7"""
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    since = time.time()

    train_loss_history = []
    val_loss_history = []
    train_acc_history = []
    val_acc_history = []

    best_val_acc = 0.0
    early_stop_counter = 0
    
    # Create a temporary directory to save training checkpoints
    with TemporaryDirectory() as tempdir:
        best_model_params_path = os.path.join(tempdir, 'best_model_params.pt')

        torch.save(model.state_dict(), best_model_params_path)
        best_acc = 0.0

        for epoch in range(num_epochs):
            print(f'Epoch {epoch}/{num_epochs - 1}')
            print('-' * 10)

            # Each epoch has a training and validation phase
            for phase in ['train', 'val']:
                if phase == 'train':
                    model.train()  # Set model to training mode
                else:
                    model.eval()   # Set model to evaluate mode

                running_loss = 0.0
                running_corrects = 0
                total_samples = 0

                # Iterate over data.
                for inputs, labels in dataloaders[phase]:
                    inputs = inputs.to(device)
                    labels = labels.to(device)

                    # zero the parameter gradients
                    optimizer.zero_grad()

                    # forward
                    # track history if only in train
                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = model(inputs)
                        _, preds = torch.max(outputs, 1)
                        loss = criterion(outputs, labels)

                        if phase == 'train' and weight_decay > 0:
                            l2_regularization = 0
                            for param in model.parameters():
                                l2_regularization += torch.norm(param, p =2)
                            loss += weight_decay * l2_regularization

                        # backward + optimize only if in training phase
                        if phase == 'train':
                            loss.backward()
                            optimizer.step()

                    # statistics
                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)
                    total_samples += inputs.size(0)

                if phase == 'train':
                    scheduler.step()

                epoch_loss = running_loss / total_samples
                epoch_acc = running_corrects.double() / total_samples

                print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

                if phase == 'val':
                    # Check for early stopping
                    if epoch_acc > best_val_acc:
                        best_val_acc = epoch_acc
                        early_stop_counter = 0
                    else:
                        early_stop_counter += 1

                if early_stop_counter >= patience:
                    print(f"Early Stopping Triggered")
                    model.load_state_dict(torch.load(best_model_params_path))
                    return model, {'train': {'loss': train_loss_history, "acc": train_acc_history},
                   "val": {"loss": val_loss_history, "acc": val_acc_history}}


                if phase == 'train':
                    train_loss_history.append(epoch_loss)
                    train_acc_history.append(epoch_acc)
                else:
                    val_loss_history.append(epoch_loss)
                    val_acc_history.append(epoch_acc)

                # deep copy the model
                if phase == 'val' and epoch_acc > best_acc:
                    best_acc = epoch_acc
                    torch.save(model.state_dict(), best_model_params_path)

            print()

        time_elapsed = time.time() - since
        print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
        print(f'Best val Acc: {best_acc:4f}')
        
        # load best model weights
        model.load_state_dict(torch.load(best_model_params_path))

    return model, {'train': {'loss': train_loss_history, "acc": train_acc_history},
                   "val": {"loss": val_loss_history, "acc": val_acc_history}}


def get_augment_transformations() -> transforms:
    return  transforms.Compose([
    transforms.Resize((256, 256)),  # Resize the image to 256x256
    transforms.RandomRotation(degrees=15),  # Random rotation up to 15 degrees
    transforms.RandomHorizontalFlip(p=0.5),  # Random horizontal flip with a probability of 0.5
    transforms.RandomResizedCrop(size=224, scale=(0.75, 1.0)),  # Random crop with a scale of 60% to 100% of the original size
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def train_resnet_architecture(dataset: Dataset, train_ratio: float, batch_size: int,
                              model: nn.Module, criterion: _Loss, optimizer_ft: Optimizer,
                              exp_lr_scheduler: _LRScheduler, patience: int, 
                              num_epochs: int, weight_decay: float = 0.0001, 
                              augment_transforms: bool = False, 
                              base_transforms: transforms = 
                                transforms.Compose([transforms.Resize((224, 224)),
                                transforms.ToTensor(),
                                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                     std=[0.229, 0.224, 0.225])])
      ) -> Tuple[nn.Module, Dict[str, Dict[str, List]]]:

    targets = [label for _, label in dataset.imgs] #get class labels

    #Use stratified sampling to get proper split
    split = StratifiedShuffleSplit(n_splits=1, test_size = 1-train_ratio, random_state=1234)
    train_indices, val_indices = next(split.split(dataset.imgs, targets))

    train_subset = Subset(dataset, train_indices)
    val_subset = Subset(dataset, val_indices)

    ##Create Dataloader Objects

    dataloaders = {
        'train': DataLoader(train_subset, batch_size = batch_size, shuffle=True),
        'val': DataLoader(val_subset, batch_size= batch_size, shuffle=False)}
    
    if augment_transforms:
        augmented_transform = get_augment_transformations()
        dataloaders['train'].dataset.transform = augmented_transform
        dataloaders['val'].dataset.transform = base_transforms

    return training_resnet_model_loop(dataloaders, model,
                criterion, optimizer_ft, exp_lr_scheduler, 
                patience, num_epochs, weight_decay)
    
def process_and_dump_training_artifacts(save_path:str, artifact_dict: dict) -> None:
    """Fixes tensor type in value list of dictionary and dumps json to save_path"""
    for outer_key in artifact_dict.keys():
        for inner_key in artifact_dict[outer_key]:
            try:
                artifact_dict[outer_key][inner_key] = [i.to("cpu") for i in artifact_dict[outer_key][inner_key]]
            except TypeError:
                continue
            except AttributeError:
                continue

def get_kfolds_indices(dataset: Dataset, kfolds: int,  train_ratio:float) -> Dict[int,List[int]]:
    """
    Get indicies for the validation dataset, all other indicies 
    will be used for training
    """

    total_data = len(dataset)
    train_size = int(train_ratio*total_data)
    indices = list(range(total_data))
    random.shuffle(indices)

    val_indices = {}
    for i in range(kfolds):
        left = int(i*total_data/kfolds)
        right = int((i + 1) * total_data / kfolds) if i != kfolds - 1 else total_data
        val_indices[i] = indices[left:right]

    return val_indices

def get_kfolds_dataset(dataset: Dataset, val_indices: list) ->  Dict[str, DataLoader]:
    """
    utilize pre-shuffled indicies to create a dataloader
    """
    total_data = len(dataset)
    indices = list(range(total_data))
    train_indices = [i for i in indices if i not in val_indices]
    random.shuffle(train_indices)


    # Create data loaders for training and testing sets
    return {'train': DataLoader(dataset, batch_size=32, sampler=train_indices),
               'val': DataLoader(dataset, batch_size=32, sampler=val_indices)}

def cross_val_model_training(dataset: Dataset, kfolds: int, train_ratio: float,
                              model: nn.Module, criterion: _Loss, optimizer_ft: Optimizer,
                              exp_lr_scheduler: _LRScheduler, patience: int, 
                              num_epochs: int, augment_transforms: bool = False, 
                              base_transforms: transforms = 
                              transforms.Compose([transforms.Resize((224, 224)),
                              transforms.ToTensor(),
                              transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                   std=[0.229, 0.224, 0.225])])
      ) -> Tuple[nn.Module, Dict[str, Dict[str, List]]]:
    """
    Train the model utilizing cross validation with kfolds
    """
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    # Define the transformations to apply to the images

    artifactDict = {}
    val_indices = get_kfolds_indices(dataset, kfolds,  train_ratio)
    best_model = model
    best_val_acc = 0

    for i in range(kfolds):
        dataloaders = get_kfolds_dataset(dataset, val_indices[i])
        if augment_transforms:
            augmented_transform = get_augment_transformations()
            dataloaders['train'].dataset.transform = augmented_transform
            dataloaders['val'].dataset.transform = base_transforms
        else:
            dataloaders['train'].dataset.transform = base_transforms
            dataloaders['val'].dataset.transform = base_transforms

        print(f"************\nTraining with kfolds {i} indicies\n************")
        model_ft, loss_acc_dict = training_resnet_model_loop(dataloaders, model, criterion, 
                                           optimizer_ft, exp_lr_scheduler, patience,
                                           num_epochs, weight_decay = 0.0001)
        if loss_acc_dict['val']['acc'][-1] > best_val_acc:
            best_model = model_ft
            best_val_acc = loss_acc_dict['val']['acc'][-1]

        artifactDict[f'kfolds {i}'] = {'training_artifacts': loss_acc_dict,
                        'val_indices': val_indices, 
                        'model_weights': model.state_dict()}
        
    return best_model, artifactDict
