import os
import copy
import time
import torch
import logging
import numpy as np
from tqdm import tqdm
from contextlib import contextmanager
from safetensors.torch import save_file, load_file
from sklearn.metrics import classification_report, balanced_accuracy_score

from transformers import Trainer
from transformers.trainer_utils import PredictionOutput, EvalPrediction, TrainOutput
from transformers.trainer_pt_utils import nested_concat, nested_numpify

class CustomTrainer(Trainer):
    def __init__(self, *args, device = torch.device('cuda' if torch.cuda.is_available() else 'cpu'),**kwargs):
        super().__init__(*args, **kwargs)
        self.model.to(device)

        # Create optimizer and scheduler
        self.create_optimizer_and_scheduler(num_training_steps=self.args.max_steps)

    def _inner_training_loop(self, batch_size, args, resume_from_checkpoint, trial, ignore_keys_for_eval):
        # Ensure model is on the correct device
        self.model.to(self.model.device)

        # Initialize tr_loss on the correct device
        tr_loss = torch.tensor(0.0).to(self.model.device)
        self.model.zero_grad()
        self.control = self.callback_handler.on_train_begin(args, self.state, self.control)
        model = self.model

        self.state.epoch = 0
        self.state.global_step = 0

        epoch_steps = len(self.get_train_dataloader())

        for epoch in range(int(args.num_train_epochs)):
            correct_predictions = 0
            total_predictions = 0
            tr_loss = torch.tensor(0.0).to(self.model.device)

            with tqdm(total=epoch_steps, desc=f"Epoch {epoch + 1}/{args.num_train_epochs}", leave=True) as pbar:
                for step, inputs in enumerate(self.get_train_dataloader()):
                    tr_loss_step = self.training_step(model, inputs)
                    tr_loss_step = tr_loss_step.to(self.model.device)  # Ensure tr_loss_step is on the correct device
                    tr_loss += tr_loss_step
                    self.state.global_step += 1

                    # Calculate accuracy
                    outputs = self.model(**inputs)
                    logits = outputs["logits"]
                    labels = inputs["labels"]
                    predictions = torch.argmax(logits, dim=-1)
                    correct_predictions += (predictions == labels).sum().item()
                    total_predictions += labels.size(0)

                    if (step + 1) % args.logging_steps == 0:
                        accuracy = correct_predictions / total_predictions
                        logs = {
                            "loss": tr_loss.item() / (step + 1),
                            "accuracy": accuracy,
                            "step": step + 1,
                            "epoch_steps": epoch_steps,
                            "epoch": epoch
                        }
                        self.log(logs)
                        pbar.set_postfix(logs)

                    if (step + 1) % args.eval_steps == 0:
                        pbar.close()  # Temporarily close the progress bar
                        eval_metrics = self.evaluate(epoch=epoch, step=step + 1)
                        pbar = tqdm(total=epoch_steps, desc=f"Epoch {epoch + 1}/{args.num_train_epochs}", leave=True)
                        pbar.n = step + 1  # Update the progress bar to the current step
                        pbar.refresh()
                        # Log evaluation metrics
                        tqdm.write(str(eval_metrics))

                    if (step + 1) % args.gradient_accumulation_steps == 0 or (
                        step + 1 == len(self.get_train_dataloader())
                    ):
                        if args.fp16 and _use_native_amp:
                            self.scaler.step(self.optimizer)
                            self.scaler.update()
                        else:
                            self.optimizer.step()
                        self.lr_scheduler.step()
                        model.zero_grad()
                        self.control = self.callback_handler.on_step_end(args, self.state, self.control)

                    pbar.update(1)

            self.state.epoch += 1
            eval_metrics = self.evaluate(epoch=epoch)
            # Log evaluation metrics
            tqdm.write(str(eval_metrics))

        self.control = self.callback_handler.on_train_end(args, self.state, self.control)
        return TrainOutput(self.state.global_step, tr_loss.item() / self.state.global_step, metrics={"train_loss": tr_loss.item() / self.state.global_step})
    
    def evaluate(self, epoch=None, step=None):
        eval_dataloader = self.get_eval_dataloader()
        self.model.eval()  # Ensure the model is in evaluation mode
    
        if len(eval_dataloader) == 0:
            raise ValueError("Evaluation dataloader is empty. Please check your evaluation dataset.")
    
        # Debug: check the first batch
        first_batch = next(iter(eval_dataloader))
        print("First batch keys:", first_batch.keys())
        print("First batch labels shape:", first_batch['labels'].shape)
        print("First batch labels unique values:", np.unique(first_batch['labels'].cpu().numpy()))
    
        if 'labels' not in first_batch:
            raise ValueError("Labels are missing in the evaluation dataset")
    
        desc = f"Evaluation{' for Epoch ' + str(epoch + 1) if epoch is not None else ''}{', Step ' + str(step) if step is not None else ''}"
        with tqdm(total=len(eval_dataloader), desc=desc, leave=True) as pbar:
            try:
                # Run prediction on a single batch first for debugging
                with torch.no_grad():
                    outputs = self.model(**first_batch)
                print("Single batch output keys:", outputs.keys())
                print("Single batch logits shape:", outputs['logits'].shape)
                print("Single batch predictions:", torch.argmax(outputs['logits'], dim=-1))
    
                # Now run the full prediction loop
                output = self.prediction_loop(
                    eval_dataloader,
                    description="Evaluation",
                    prediction_loss_only=False,
                    ignore_keys=None,
                    metric_key_prefix="eval"
                )
            except Exception as e:
                print(f"Error during prediction: {str(e)}")
                raise
    
            pbar.update(len(eval_dataloader))
    
        print("Evaluation Output:", output)
        if self.compute_metrics is not None:
            print("Compute Metrics Input shapes:")
            print("  Predictions shape:", output.predictions.shape)
            print("  Labels shape:", output.label_ids.shape)
            print("Unique values in predictions:", np.unique(np.argmax(output.predictions, axis=-1)))
            print("Unique values in labels:", np.unique(output.label_ids))
            metrics = self.compute_metrics(EvalPrediction(predictions=output.predictions, label_ids=output.label_ids))
            print("Computed Metrics:", metrics)
        else:
            metrics = {}
        
        self.log(metrics)
        self.control = self.callback_handler.on_evaluate(self.args, self.state, self.control, metrics)
    
        return metrics

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        args = self.args
        
        inputs = self._prepare_inputs(inputs)
        with torch.no_grad():
            outputs = model(**inputs)
        
        if isinstance(outputs, dict):
            loss = outputs.get("loss")
            logits = outputs.get("logits")
        else:
            loss, logits = outputs[:2]
        
        # Ensure loss is a scalar tensor
        if loss is not None:
            loss = loss.mean().detach()
        else:
            loss = torch.tensor(0.0).to(self.args.device)
        
        # If prediction_loss_only is True, just return the loss
        if prediction_loss_only:
            return (loss, None, None)
        
        # Get the labels
        labels = inputs.get("labels")
        
        # Keep logits and labels as tensors, don't convert to numpy
        if logits is not None:
            logits = logits.detach()
        
        if labels is not None:
            labels = labels.detach()
        
        # Return a tuple of (loss, logits, labels)
        return (loss, logits, labels)
            
    def prediction_loop(self, dataloader, description, prediction_loss_only=None, ignore_keys=None, metric_key_prefix="eval"):
        args = self.args
    
        if prediction_loss_only is None:
            prediction_loss_only = args.prediction_loss_only
    
        model = self.model
    
        # Use the batch size from training arguments
        batch_size = args.per_device_eval_batch_size
        num_examples = self.num_examples(dataloader)
    
        model.eval()
    
        preds_host = None
        labels_host = None
        losses_host = None
    
        world_size = max(1, args.world_size)
    
        for step, inputs in enumerate(dataloader):
            loss, logits, labels = self.prediction_step(model, inputs, prediction_loss_only, ignore_keys=ignore_keys)
            
            if loss is not None:
                losses = loss.repeat(batch_size)
                losses_host = losses if losses_host is None else torch.cat((losses_host, losses), dim=0)
            if logits is not None:
                preds_host = logits if preds_host is None else nested_concat(preds_host, logits, padding_index=-100)
            if labels is not None:
                labels_host = labels if labels_host is None else nested_concat(labels_host, labels, padding_index=-100)
        
        if args.past_index and hasattr(self, "_past"):
            del self._past
    
        # Convert to NumPy arrays here
        if preds_host is not None:
            preds = nested_numpify(preds_host)
        else:
            preds = None
    
        if labels_host is not None:
            labels = nested_numpify(labels_host)
        else:
            labels = None
    
        if losses_host is not None:
            losses = nested_numpify(losses_host)
        else:
            losses = None
    
        if self.compute_metrics is not None and preds is not None and labels is not None:
            metrics = self.compute_metrics(EvalPrediction(predictions=preds, label_ids=labels))
        else:
            metrics = {}
    
        if losses is not None:
            metrics[f"{metric_key_prefix}_loss"] = losses.mean().item()
    
        return PredictionOutput(predictions=preds, label_ids=labels, metrics=metrics)
    
    def log(self, logs):
        logs["epoch"] = self.state.epoch
        logs["step"] = self.state.global_step
        self.state.log_history.append(logs)
        self.control = self.callback_handler.on_log(self.args, self.state, self.control, logs)
        if self.is_world_process_zero():  # Ensure logging only happens in the main process
            tqdm.write(str(logs))  # Use tqdm.write to avoid messing up the progress bar


class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint', trace_func=print):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = float('inf')
        self.delta = delta
        self.path = path
        self.trace_func = trace_func

    def __call__(self, val_loss, model):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
                return True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0
        return False

    def save_checkpoint(self, val_loss, model):
        if self.verbose:
            self.trace_func(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}). Saving model ...')
        try:
            model.save_pretrained(self.path, safe_serialization=True)
        except Exception as e:
            self.trace_func(f"Error saving model: {e}")
        self.val_loss_min = val_loss

def evaluate(model, dataloader, criterion, device, target_class=None, target_metric_name='f1', weighted=False):
    """
    Evaluate the model on the given dataloader.

    Args:
        model: The model to evaluate.
        dataloader: DataLoader for evaluation data.
        criterion: The loss function.
        device: The device to run the model on (e.g., 'cuda' or 'cpu').
        target_class (int, optional): The target class for optimization. If None, uses macro average.
        target_metric_name (str): The name of the target metric ('f1' or 'accuracy').
        weighted (bool): Whether to use weighted metrics for multi-class optimization.

    Returns:
        tuple: (avg_loss, macro_accuracy, macro_f1, weighted_f1, balanced_accuracy, class_metrics, target_metric)
    """
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            outputs = model(**batch)
            loss = criterion(outputs['logits'], batch['labels'])
            
            total_loss += loss.item()
            preds = torch.argmax(outputs['logits'], dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(batch['labels'].cpu().numpy())
    
    # Compute average loss
    avg_loss = total_loss / len(dataloader)
    
    # Generate classification report
    report = classification_report(all_labels, all_preds, output_dict=True)
    
    # Extract metrics
    macro_accuracy = report['accuracy']
    macro_f1 = report['macro avg']['f1-score']
    weighted_f1 = report['weighted avg']['f1-score']
    balanced_accuracy = balanced_accuracy_score(all_labels, all_preds)
    
    # Compute class-specific metrics
    class_metrics = {f"class_{i}": {"accuracy": report[str(i)]['precision'], "f1": report[str(i)]['f1-score']} 
                     for i in range(len(report) - 3)}  # -3 to exclude 'accuracy', 'macro avg', and 'weighted avg'
    
    # Compute target metric
    if target_class is None:
        # Use a combination of metrics
        target_metric = (macro_f1 + weighted_f1 + balanced_accuracy) / 3
    else:
        if weighted:
            # Multi-objective optimization
            class_weight = 0.75  # Adjust this weight as needed
            class_score = class_metrics[f"class_{target_class}"][target_metric_name]
            overall_score = (macro_f1 + weighted_f1 + balanced_accuracy) / 3
            target_metric = class_weight * class_score + (1 - class_weight) * overall_score
        else:
            # Single class optimization
            target_metric = class_metrics[f"class_{target_class}"][target_metric_name]
    
    return avg_loss, macro_accuracy, macro_f1, weighted_f1, balanced_accuracy, class_metrics, target_metric

@contextmanager
def evaluation_mode(model):
    """
    Context manager to temporarily set the model to evaluation mode.

    Args:
        model: The model to set to evaluation mode.

    Yields:
        None
    """
    model.eval()  # Set model to evaluation mode
    yield
    model.train()  # Set model back to training mode
    
def train_epoch(model, dataloader, optimizer, criterion, device):
    """
    Train the model for one epoch and return loss and accuracy.
    """
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    for batch in tqdm(dataloader, desc="Training"):
        optimizer.zero_grad()
        outputs = model(**batch)
        loss = criterion(outputs['logits'], batch['labels'])
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        _, predicted = torch.max(outputs['logits'], 1)
        total += batch['labels'].size(0)
        correct += (predicted == batch['labels']).sum().item()
    
    avg_loss = total_loss / len(dataloader)
    accuracy = correct / total
    return avg_loss, accuracy

def train_model(model, num_epochs, train_dataloader, eval_dataloader,
                optimizer, criterion, device, target_class=None, target_metric_name='f1',
                early_stopping=True, verbose=True, save_dir='.', weighted=False):
    """
    Train a model with the given parameters and return the trained model and training history.
    """
    
    def setup_directories():
        os.makedirs(save_dir, exist_ok=True)
        return os.path.join(save_dir, 'best_model'), os.path.join(save_dir, 'final_model')

    def setup_logging():
        logger = logging.getLogger()
        is_logging = logger.hasHandlers()
        return lambda msg: logger.info(msg) if is_logging else print(msg) if verbose else None

    def update_history(history, metrics):
        for name, value in metrics.items():
            if name not in history:
                history[name] = []
            history[name].append(value)

    def log_epoch_results(epoch, train_metrics, eval_metrics, class_metrics):
        log_or_print(f"Epoch {epoch+1}/{num_epochs} completed")
        log_or_print(f"Training - Loss: {train_metrics['loss']:.4f}, Accuracy: {train_metrics['accuracy']:.4f}")
        log_or_print(f"Validation - Loss: {eval_metrics['loss']:.4f}, Accuracy: {eval_metrics['accuracy']:.4f}")
        for name, value in eval_metrics.items():
            if name not in ['loss', 'accuracy']:
                log_or_print(f"{name.capitalize().replace('_', ' ')}: {value:.4f}")
        for class_name, class_metric in class_metrics.items():
            log_or_print(f"{class_name} - Accuracy: {class_metric['accuracy']:.4f}, F1: {class_metric['f1']:.4f}")

    def save_model(model, path, message):
        try:
            model.save_pretrained(path, safe_serialization=True)
            log_or_print(message)
        except Exception as e:
            log_or_print(f"Error saving model: {e}")

    best_model_dir, final_model_dir = setup_directories()
    log_or_print = setup_logging()
    
    minimize_metric = target_metric_name.lower() in ['loss']
    best_target_metric = float('inf') if minimize_metric else float('-inf')
    history = {}
    
    early_stopper = EarlyStopping(patience=5, verbose=verbose, path=best_model_dir) if early_stopping else None

    log_or_print(f"Starting training for {num_epochs} epochs")
    log_or_print(f"Target class: {target_class}, Target metric: {target_metric_name}")
    log_or_print(f"Weighted: {weighted}, Early stopping: {early_stopping}")

    start_time = time.time()
    for epoch in range(num_epochs):
        epoch_start_time = time.time()
        log_or_print(f"Starting epoch {epoch+1}/{num_epochs}")

        # Train for one epoch
        train_loss, train_accuracy = train_epoch(model, train_dataloader, optimizer, criterion, device)
        
        # Evaluate the model
        with evaluation_mode(model):
            eval_results = evaluate(model, eval_dataloader, criterion, device, target_class, target_metric_name, weighted)
        
        eval_loss, accuracy, macro_f1, weighted_f1, balanced_accuracy, class_metrics, target_metric = eval_results
        
        # Collect metrics
        train_metrics = {'loss': train_loss, 'accuracy': train_accuracy}
        eval_metrics = {
            'loss': eval_loss,
            'accuracy': accuracy,
            'macro_f1': macro_f1,
            'weighted_f1': weighted_f1,
            'balanced_accuracy': balanced_accuracy,
            'target_metric': target_metric
        }
        
        # Update history and log results
        update_history(history, {**train_metrics, **eval_metrics})
        log_epoch_results(epoch, train_metrics, eval_metrics, class_metrics)
        
        # Log target metric
        target_desc = f"{'Weighted ' if weighted else ''}{'Macro' if target_class is None else f'Class {target_class}'} {target_metric_name}"
        log_or_print(f"Target Metric ({target_desc}): {target_metric:.4f}")
        
        # Check if this is the best model so far
        is_best = (minimize_metric and target_metric < best_target_metric) or \
                  (not minimize_metric and target_metric > best_target_metric)
        
        if is_best:
            best_target_metric = target_metric
            save_model(model, best_model_dir, f"New best model saved with target metric: {best_target_metric:.4f}")
        else:
            log_or_print(f"Target metric did not improve. Best: {best_target_metric:.4f}")
        
        # Early stopping check
        if early_stopper and early_stopper(eval_loss, model):
            log_or_print("Early stopping triggered")
            break

        epoch_end_time = time.time()
        log_or_print(f"Epoch {epoch+1} completed in {epoch_end_time - epoch_start_time:.2f} seconds")

    total_time = time.time() - start_time
    log_or_print(f"Training completed in {total_time:.2f} seconds")

    # Load the best model
    try:
        model = type(model).from_pretrained(best_model_dir, safe_serialization=True).to(device)
        log_or_print("Loaded best model for final evaluation")
    except Exception as e:
        log_or_print(f"Error loading best model: {e}. Continuing with the current model state.")

    # Final evaluation
    log_or_print("Starting final evaluation")
    with evaluation_mode(model):
        final_results = evaluate(model, eval_dataloader, criterion, device, target_class, target_metric_name, weighted)
    
    log_or_print("Final Evaluation Results:")
    final_metrics = dict(zip(['Loss', 'Accuracy', 'Macro F1', 'Weighted F1', 'Balanced Accuracy'], final_results[:5]))
    log_epoch_results(num_epochs, {'loss': 0, 'accuracy': 0}, final_metrics, final_results[-2])

    # Save the final model
    save_model(model, final_model_dir, "Final model saved")

    return model, history