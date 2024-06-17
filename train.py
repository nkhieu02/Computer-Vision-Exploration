import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import wandb
import metrics

def train(model: nn.Module, 
          criterion: nn.Module, 
          optimizer: nn.Module,
            dataloader: DataLoader,
            device: str):
    running_loss = 0.0
    for _, data in enumerate(dataloader):
        inputs, labels = data
        inputs = inputs.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        print(f"Training Loss: {loss.item():.4f}")
    running_loss = running_loss / len(dataloader)
    # wandb.log({"train_loss": running_loss})
    print(f"Training Loss: {running_loss:.4f}")
    return running_loss

def test(model: nn.Module,
          criterion: nn.Module, 
          dataloader: DataLoader, 
          categories: list, 
          prefix: str,
          device : str):
    all_labels = []
    all_logits = []
    running_loss = 0.0
    with torch.no_grad():
        for _, data in enumerate(dataloader):
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            all_labels.append(labels)
            all_logits.append(outputs)
    
    running_loss = running_loss / len(dataloader)
    all_labels_tensor = torch.concat(all_labels)
    all_logits_tensor = torch.concat(all_logits)
    f1_scores, precisions, recalls, accuracies, roc_auc_scores = \
        metrics.classification_metrics_n_class(categories, all_logits_tensor, all_labels_tensor, prefix= prefix)
    # wandb.log({"test_loss": running_loss})
    # wandb.log(f1_scores)
    # wandb.log(precisions)
    # wandb.log(recalls)
    # wandb.log(accuracies)
    # wandb.log(roc_auc_scores)
    print(f"Testing Loss: {running_loss:.4f}")
    return running_loss, f1_scores, precisions, recalls, accuracies, roc_auc_scores
