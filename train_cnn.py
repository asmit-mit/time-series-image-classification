import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os
from collections import Counter

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BASE_PATH = "datasets/Earthquakes/augmented/images"
OUTPUT_DIR = "train/Earthquakes/cnn"


class Model(nn.Module):
    def __init__(self, in_channels=1, num_classes=2):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, 32, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)

        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)

        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)

        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2)

        self.dropout_conv = nn.Dropout2d(0.2)
        self.dropout_fc = nn.Dropout(0.5)

        self.gap = nn.AdaptiveAvgPool2d((1, 1))

        self.fc1 = nn.Linear(128, 256)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.pool(self.relu(self.bn1(self.conv1(x))))

        x = self.pool(self.relu(self.bn2(self.conv2(x))))
        x = self.dropout_conv(x)

        x = self.pool(self.relu(self.bn3(self.conv3(x))))
        x = self.dropout_conv(x)

        x = self.gap(x)
        x = torch.flatten(x, 1)

        x = self.relu(self.fc1(x))
        x = self.dropout_fc(x)
        x = self.fc2(x)

        return x


def evaluate(model, loader):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(images)
            _, predicted = torch.max(outputs, 1)

            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    return correct / total if total > 0 else 0.0


def get_transform_and_channels(method_name):
    if method_name == "combo":
        transform = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        in_channels = 3
    else:
        transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        in_channels = 1

    return transform, in_channels


def train_one_method(method_name):
    print(f"\n=== Training on {method_name} ===")

    transform, in_channels = get_transform_and_channels(method_name)

    train_data = datasets.ImageFolder(
        os.path.join(BASE_PATH, method_name, "train"),
        transform=transform
    )

    test_data = datasets.ImageFolder(
        os.path.join(BASE_PATH, method_name, "test"),
        transform=transform
    )

    train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=32)

    class_counts = Counter(train_data.targets)
    print("Class distribution:", class_counts)

    total = sum(class_counts.values())

    weights = torch.tensor(
        [total / class_counts[c] for c in sorted(class_counts.keys())],
        dtype=torch.float
    ).to(DEVICE)

    print("Class weights:", weights)

    model = Model(in_channels=in_channels).to(DEVICE)

    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    EPOCHS = 80

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    log_path = os.path.join(OUTPUT_DIR, f"{method_name}_log.txt")
    model_path = os.path.join(OUTPUT_DIR, f"{method_name}_best.pth")

    best_acc = 0
    best_epoch = 0

    with open(log_path, "w") as log_file:
        log_file.write("epoch,loss,train_acc,test_acc\n")

        for epoch in range(EPOCHS):
            model.train()
            total_loss = 0

            for images, labels in train_loader:
                images, labels = images.to(DEVICE), labels.to(DEVICE)

                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            avg_loss = total_loss / len(train_loader)

            train_acc = evaluate(model, train_loader)
            test_acc = evaluate(model, test_loader)

            if test_acc > best_acc:
                best_acc = test_acc
                best_epoch = epoch + 1
                torch.save(model.state_dict(), model_path)

            log_file.write(f"{epoch + 1},{avg_loss:.4f},{train_acc:.4f},{test_acc:.4f}\n")

            if (epoch + 1) % 5 == 0:
                print(
                    f"Epoch {epoch + 1}: "
                    f"Loss={avg_loss:.4f}, "
                    f"Train={train_acc:.4f}, "
                    f"Test={test_acc:.4f}"
                )

    final_acc = evaluate(model, test_loader)

    summary_path = os.path.join(OUTPUT_DIR, f"{method_name}_summary.txt")
    with open(summary_path, "w") as f:
        f.write(f"Final Accuracy: {final_acc:.4f}\n")
        f.write(f"Best Accuracy: {best_acc:.4f}\n")
        f.write(f"Best Epoch: {best_epoch}\n")

    print(f"\nFinal Test Accuracy ({method_name}): {final_acc:.4f}")
    print(f"Best Accuracy: {best_acc:.4f} at epoch {best_epoch}")


def main():
    methods = ["gaf", "gadf", "rp", "combo"]

    for method in methods:
        train_one_method(method)

    print("\nTraining complete.")


if __name__ == "__main__":
    main()
