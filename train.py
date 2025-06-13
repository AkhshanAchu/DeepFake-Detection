import torch
import torch.nn as nn
from tqdm import tqdm
from models import classifier_block
from utils import get_data_loaders, count_parameters, plot_training_metrics, save_model


def train_model():
    batch_size = 32
    learning_rate = 1e-4
    num_epochs = 100
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_path = r"C:\Users\shivn\Documents\SP Cup\Dataset"
    test_path = r"C:\Users\shivn\Documents\SP Cup\validation"
    
    train_loader, val_loader, test_loader = get_data_loaders(
        train_path, test_path, batch_size=batch_size, max_train_samples=100000
    )

    model = classifier_block(n_blocks=1).to(device)
    loss_fn = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    print(f"Train Size: {len(train_loader)} | Val Length: {len(val_loader)} | Test Length: {len(test_loader)}")
    
    num_params = count_parameters(model)
    print(f"Total number of parameters: {num_params}")

    best_val_acc = 0.0
    train_losses, val_losses = [], []
    train_accuracies, val_accuracies = [], []

    for epoch in range(num_epochs):
        model.train()
        running_train_loss = 0.0
        correct_train = 0
        total_train = 0

        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} - Training", leave=False):
            images, labels = batch
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()

            outputs = model(images)

            loss = loss_fn(outputs, labels)

            loss.backward()
            optimizer.step()

            running_train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            correct_train += (predicted == labels).sum().item()
            total_train += labels.size(0)

        avg_train_loss = running_train_loss / len(train_loader)
        train_accuracy = correct_train / total_train

        model.eval()
        running_val_loss = 0.0
        correct_val = 0
        total_val = 0

        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} - Validation", leave=False):
                images, labels = batch
                images, labels = images.to(device), labels.to(device)

                outputs = model(images)

                val_loss = loss_fn(outputs, labels)
                running_val_loss += val_loss.item()

                _, predicted = torch.max(outputs.data, 1)
                correct_val += (predicted == labels).sum().item()
                total_val += labels.size(0)

        avg_val_loss = running_val_loss / len(val_loader)
        val_accuracy = correct_val / total_val

        running_test_loss = 0.0
        correct_test = 0
        total_test = 0

        with torch.no_grad():
            for batch in tqdm(test_loader, desc=f"Epoch {epoch+1}/{num_epochs} - Testing", leave=False):
                images, labels = batch
                images, labels = images.to(device), labels.to(device)

                outputs = model(images)

                test_loss = loss_fn(outputs, labels)
                running_test_loss += test_loss.item()

                _, predicted = torch.max(outputs.data, 1)
                correct_test += (predicted == labels).sum().item()
                total_test += labels.size(0)

        avg_test_loss = running_test_loss / len(test_loader)
        test_accuracy = correct_test / total_test

        if val_accuracy > best_val_acc:
            best_val_acc = val_accuracy
            save_model(model, 'best_model.pth')

        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        train_accuracies.append(train_accuracy)
        val_accuracies.append(val_accuracy)

        print(f"Epoch {epoch+1}/{num_epochs}: Train Loss: {avg_train_loss:.4f}, Train Accuracy: {train_accuracy*100:.2f}% | "
              f"Val Loss: {avg_val_loss:.4f}, Val Accuracy: {val_accuracy*100:.2f}% | "
              f"Test Loss: {avg_test_loss:.4f}, Test Accuracy: {test_accuracy*100:.2f}%")

    plot_training_metrics(train_losses, val_losses, train_accuracies, val_accuracies, num_epochs)


if __name__ == "__main__":
    train_model()