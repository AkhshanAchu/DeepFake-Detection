import torch
import matplotlib.pyplot as plt


def count_parameters(model):
    total_params = sum(p.numel() for p in model.parameters())
    return total_params


def plot_training_metrics(train_losses, val_losses, train_accuracies, val_accuracies, num_epochs):
    epochs = range(1, num_epochs + 1)

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, label='Train Loss')
    plt.plot(epochs, val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Train and Validation Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_accuracies, label='Train Accuracy')
    plt.plot(epochs, val_accuracies, label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Train and Validation Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.show()


def save_model(model, filepath):
    torch.save(model.state_dict(), filepath)


def load_model(model, filepath, device):
    model.load_state_dict(torch.load(filepath, map_location=device))
    return model
