import torch
import torch.nn as nn
import torch.optim as optim

"""
Function: train  
This function trains a given model using a specified optimizer and loss function.  
It iterates through multiple epochs, computing loss and evaluation metrics for both the training and validation datasets.  
During training, it updates model parameters using backpropagation, and during validation, it evaluates model performance without updating weights.  
The function returns the recorded losses and metrics for further analysis.  
"""

def train(model,device, train_loader, valid_loader, epochs, metric, optimizer, loss_funtion):
    model.to(device)
    
    train_losses, valid_losses = [] , []
    train_metrics, valid_metrics = [], [] #Lists to store training and validation losses/metrics for each epoch.

    for epoch in range(epochs):
        model.train() # This puts the model into training mode
        running_loss, running_metric = 0.0, 0.0 #Initializes running loss and metric to 0.0.
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            #  The gradients themselves should not accumulate across batches—only across epochs through updated weights.
            # so we ahve to clears gradient before backpropagation
            optimizer.zero_grad() 
            outputs = model(inputs) # Passes the inputs through the model
            loss = loss_funtion(outputs, labels) #Compute loss between predictions and true labels
            # Backprpagation and optimization
            loss.backward() # Computs gradient through back proapagation
            optimizer.step() # Updates model parameters

            running_loss += loss.item()
            running_metric += metric(outputs, labels).item() # Accumulates total loss and metric for the epoch

        train_loss = running_loss / len(train_loader) # Computes the average loss over all batches.
        train_metric = running_metric / len(train_loader)  # Computes the average metric over all batches.
        train_losses.append(train_loss) # Stores the values in the tracking lists.
        train_metrics.append(train_metric) # Stores the values in the tracking lists.

        model.eval() # Puts the model in evaluation mode (disables dropout, etc.).
        running_loss, running_metric = 0.0, 0.0 #Resets validation loss and metric.

        with torch.no_grad():
            for inputs, labels in valid_loader:  # loop through validation dataset
                inputs, labels = inputs.to(device), labels.to(device) # moving data to the correct device and compute prediction
                outputs = model(inputs)
                loss = loss_funtion(outputs, labels)

                # Accumulates validation loss and metric.
                running_loss  += loss.item() 
                running_metric += metric(outputs, labels).item()  
        # Computes the average validation loss and metric.Stores them in lists.
        valid_loss = running_loss / len(valid_loader)
        valid_metric = running_metric / len(valid_loader)
        valid_losses.append(valid_loss)
        valid_metrics.append(valid_metric)

        print(f"Epoch {epoch+1}/{epochs}: Train Loss: {train_loss:.4f}, Valid Loss: {valid_loss:.4f}, Train Metric: {train_metric:.4f}, Valid Metric: {valid_metric:.4f}")


    return train_losses, valid_losses, train_metrics, valid_metrics, model

