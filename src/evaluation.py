import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import time
import torch
import torch.nn as nn

class Evaluation:
    
    def evaluate_model(self):
        self.model.eval()
        y_true, y_pred = [], []
        total_loss = 0.0
        total_time = 0.0
        criterion = nn.CrossEntropyLoss()
        
        with torch.no_grad():
            for images, labels in self.test_loader:
                start_time = time.time()
                outputs = self.model(images)
                end_time = time.time()
                
                loss = criterion(outputs, labels)
                total_loss += loss.item()
                total_time += (end_time - start_time)
                
                _, predicted = torch.max(outputs, 1)
                y_true.extend(labels.numpy())
                y_pred.extend(predicted.numpy())
        
        accuracy = accuracy_score(y_true, y_pred)
        avg_loss = total_loss / len(self.test_loader)
        avg_time = total_time / len(self.test_loader)
        
        print(f"Test Accuracy: {accuracy:.4f}")
        print(f"Average Loss: {avg_loss:.4f}")
        print(f"Average Inference Time per Batch: {avg_time:.4f} sec")
        print("Classification Report:")
        print(classification_report(y_true, y_pred))
        
        return y_true, y_pred, accuracy, avg_loss, avg_time
    
    def plot_confusion_matrix(self, y_true, y_pred):
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(10, 7))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.title("Confusion Matrix")
        plt.show()

    
    def plot_loss_accuracy(self, accuracy, avg_loss):
        metrics = ['Accuracy', 'Average Loss']
        values = [accuracy, avg_loss]
        
        plt.figure(figsize=(8, 5))
        plt.bar(metrics, values, color=['blue', 'red'])
        plt.ylabel("Value")
        plt.title("Model Performance Metrics")
        plt.show()

'''For seperate plotting for the graphs'''

    # def plot_accuracy(self, accuracy):
    #     plt.figure(figsize=(6, 4))
    #     plt.bar(['Accuracy'], [accuracy], color='blue')
    #     plt.ylabel("Value")
    #     plt.title("Model Accuracy")
    #     plt.ylim(0, 1)
    #     plt.show()
    
    # def plot_loss(self, avg_loss):
    #     plt.figure(figsize=(6, 4))
    #     plt.bar(['Average Loss'], [avg_loss], color='red')
    #     plt.ylabel("Value")
    #     plt.title("Model Loss")
    #     plt.show()