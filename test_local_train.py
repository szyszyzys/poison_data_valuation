import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from marketplace.utils.gradient_market_utils.data_processor import get_data_set
from model.utils import get_model


# -------------------------------
# Training Function (provided)
# -------------------------------
def train_local_model(model: nn.Module,
                      train_loader: DataLoader,
                      criterion: nn.Module,
                      optimizer: optim.Optimizer,
                      device: torch.device,
                      epochs: int = 1) -> nn.Module:
    """
    Train the model on the given train_loader for a specified number of epochs.
    """
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for batch_data, batch_labels in train_loader:
            batch_data = batch_data.to(device)
            batch_labels = batch_labels.to(device)
            optimizer.zero_grad()
            outputs = model(batch_data)
            loss = criterion(outputs, batch_labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * batch_data.size(0)
        avg_loss = running_loss / len(train_loader.dataset)
        print(f"Epoch [{epoch + 1}/{epochs}], Loss: {avg_loss:.4f}")
    return model


# -------------------------------
# Define a Simple Classification Model
# -------------------------------
class SimpleClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes):
        super(SimpleClassifier, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)  # output logits for CrossEntropyLoss (do not apply softmax here)
        return x


# -------------------------------
# Create Synthetic Dataset
# -------------------------------
def create_synthetic_data(num_samples=1000, input_dim=20, num_classes=4):
    """
    Generates synthetic data for a classification task.
    Features are sampled from a normal distribution, and labels are random integers.
    """
    X = torch.randn(num_samples, input_dim)
    y = torch.randint(0, num_classes, (num_samples,))
    return X, y


# -------------------------------
# Full Pipeline
# -------------------------------
def main():
    # Hyperparameters
    input_dim = 20  # Number of input features
    hidden_dim = 64  # Size of the hidden layer
    num_classes = 4  # Number of classes
    num_samples = 1000  # Number of synthetic data samples
    batch_size = 32
    learning_rate = 0.001
    num_epochs = 5
    dataset_name = "FMNIST"
    client_loaders, full_dataset = get_data_set(dataset_name, num_clients=11, iid=True)

    # config the buyer
    # buyer = GradientSeller(seller_id="buyer", local_data=client_loaders[buyer_cid].dataset, dataset_name=dataset_name,
    #                        save_path=save_path)
    train_dataset = client_loaders[0].dataset
    # Create synthetic dataset and DataLoader
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # Set device (GPU if available, else CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize model, loss function, and optimizer
    model = get_model("FMNIST")
    # Load base parameters into the model
    model = model.to("cuda:6")

    criterion = nn.CrossEntropyLoss()  # expects raw logits and integer class labels
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Train the model
    print("Starting training...\n")
    model = train_local_model(model, train_loader, criterion, optimizer, device, epochs=num_epochs)

    # Evaluate the model on the training set
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_data, batch_labels in train_loader:
            batch_data = batch_data.to(device)
            batch_labels = batch_labels.to(device)
            outputs = model(batch_data)
            _, predicted = torch.max(outputs, dim=1)
            total += batch_labels.size(0)
            correct += (predicted == batch_labels).sum().item()

    accuracy = correct / total * 100
    print(f"\nTraining Accuracy: {accuracy:.2f}%")


if __name__ == '__main__':
    main()
