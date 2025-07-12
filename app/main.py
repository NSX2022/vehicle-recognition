import sys
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
import timm
from PIL import Image
from glob import glob

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from tqdm import tqdm

#Change per model, b0 = 224, b1 = 240, b2 = 260, b3 = 300, b4 = 380, b5 = 456, b6 = 528, b7 = 600
IMAGE_WIDTH_HEIGHT = 300
BATCH_SIZE_PER_MODEL = 64


class ABDataSet(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data = ImageFolder(data_dir, transform)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    @property
    def classes(self):
        return self.data.classes


class SimpleClassifier(nn.Module):
    def __init__(self, num_classes=3):
        super(SimpleClassifier, self).__init__()
        self.base_model = timm.create_model('efficientnet_b3', pretrained=True)

        enet_out_size = self.base_model.classifier.in_features

        #Remove the original classifier
        self.base_model.classifier = nn.Identity()

        #Custom classifier
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(enet_out_size, enet_out_size // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(enet_out_size // 2, num_classes)
        )

    def forward(self, x):
        features = self.base_model(x)
        output = self.classifier(features)
        return output


def sysinfo():
    print('System Version: ' + sys.version)
    print('Pytorch Version: ' + torch.__version__)
    print('Torchvision Version: ' + torchvision.__version__)
    print('Numpy Version: ' + np.__version__)
    print('Pandas Version: ' + pd.__version__)
    print(f"CUDA available: {torch.cuda.is_available()}")
    has_cuda = torch.cuda.is_available()

    if has_cuda:
        print(f"CUDA version: {torch.version.cuda}")
        print(f"Number of GPUs: {torch.cuda.device_count()}")
        print(f"GPU name: {torch.cuda.get_device_name(0)}")
        print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024 ** 3:.1f} GB")
    return


#transform variation
train_transform = transforms.Compose([
    transforms.Resize((IMAGE_WIDTH_HEIGHT, IMAGE_WIDTH_HEIGHT)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

val_test_transform = transforms.Compose([
    transforms.Resize((IMAGE_WIDTH_HEIGHT, IMAGE_WIDTH_HEIGHT)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def main():
    sysinfo()

    train_dir = './train/'
    valid_dir = './valid/'
    test_dir = './test/'

    training_dataset = ABDataSet(train_dir, train_transform)
    valid_dataset = ABDataSet(valid_dir, val_test_transform)
    test_dataset = ABDataSet(test_dir, val_test_transform)

    train_loader = DataLoader(training_dataset, batch_size=BATCH_SIZE_PER_MODEL, shuffle=True)
    val_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE_PER_MODEL, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE_PER_MODEL, shuffle=False)

    num_epochs = int(input('Enter number of Epochs: '))
    num_tests = int(input('Enter number of Tests: '))
    response = input('Replace when testing? (True/False, default=True): ').strip().lower()
    do_replace = response not in ['false', 'f', 'no', 'n', '0', 'x']
    visualize = input('Open a window to show each test prediction? (True/False, default=True): ').strip().lower()
    do_visualize = visualize not in ['false', 'f', 'no', 'n', '0', 'x']

    train_losses, val_losses = [], []
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = SimpleClassifier(num_classes=len(training_dataset.classes))

    # Load existing weights if available
    if os.path.exists('classifier_weights.pth'):
        model.load_state_dict(torch.load('classifier_weights.pth', map_location=device))
        print("Loaded existing weights")
    else:
        print("No existing weights found, starting with pretrained EfficientNet weights")

    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    #Training loop
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for images, labels in tqdm(train_loader, desc=f'Training Epoch {epoch + 1}/{num_epochs}'):
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * labels.size(0)

        scheduler.step()
        train_loss = running_loss / len(train_loader.dataset)
        train_losses.append(train_loss)

        # Validation phase
        model.eval()
        running_loss = 0.0
        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc=f'Validation Epoch {epoch + 1}/{num_epochs}'):
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                running_loss += loss.item() * labels.size(0)

        val_loss = running_loss / len(val_loader.dataset)
        val_losses.append(val_loss)
        print(f"Epoch {epoch + 1}/{num_epochs} - Train loss: {train_loss:.4f}, Validation loss: {val_loss:.4f}")

    torch.save(model.state_dict(), 'classifier_weights.pth')
    print("Saved state dictionary")

    #Plot training curves
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training loss')
    plt.plot(val_losses, label='Validation loss')
    plt.legend()
    plt.title("Loss over epochs")
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True, alpha=0.3)
    plt.show()

    test_images = glob('./test/*/*')

    #Limit num_tests when not replacing to avoid the dreaded overflow error
    if not do_replace:
        num_tests = min(num_tests, len(test_images))
        print(f"Warning: num_tests reduced to {num_tests} (total available test images)")

    test_examples = np.random.choice(test_images, num_tests, replace=do_replace)

    #Accuracy
    correct_predictions = 0
    total_predictions = 0
    accuracies = []

    for i, example in enumerate(test_examples):
        original_image, image_tensor = preprocess_image(example, val_test_transform)
        probabilities = predict(model, image_tensor, device)

        predicted_class = np.argmax(probabilities)

        #Get correct class from file path
        true_class_name = os.path.basename(os.path.dirname(example))
        true_class = training_dataset.classes.index(true_class_name)

        if predicted_class == true_class:
            correct_predictions += 1
        total_predictions += 1

        #Calculate running accuracy
        current_accuracy = correct_predictions / total_predictions
        accuracies.append(current_accuracy)

        class_names = training_dataset.classes
        if do_visualize:
            visualize_predictions(original_image, probabilities, class_names)

    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(accuracies) + 1), accuracies, 'b-', linewidth=2)
    plt.axhline(y=np.mean(accuracies), color='r', linestyle='--',
                label=f'Final Accuracy: {np.mean(accuracies):.2%}')
    plt.xlabel('Test Sample Number')
    plt.ylabel('Accuracy')
    plt.title('Model Accuracy During Testing')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

    print(
        f"Final Test Accuracy: {np.mean(accuracies):.2%}")

def preprocess_image(image_path, transform):
    image = Image.open(image_path).convert("RGB")
    return image, transform(image).unsqueeze(0)

def predict(model, image_tensor, device):
    model.eval()
    with torch.no_grad():
        image_tensor = image_tensor.to(device)
        outputs = model(image_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
    return probabilities.cpu().numpy().flatten()


#Visualization, stats
def visualize_predictions(original_image, probabilities, class_names):
    fig, axarr = plt.subplots(1, 2, figsize=(14, 7))

    axarr[0].imshow(original_image)
    axarr[0].axis("off")
    axarr[0].set_title("Input Image")

    colors = ['red' if p == max(probabilities) else 'skyblue' for p in probabilities]
    axarr[1].barh(class_names, probabilities, color=colors)
    axarr[1].set_xlabel("Probability")
    axarr[1].set_title("Class Predictions")
    axarr[1].set_xlim(0, 1)

    for i, (name, prob) in enumerate(zip(class_names, probabilities)):
        axarr[1].text(prob + 0.01, i, f'{prob:.2%}',
                      va='center', fontweight='bold')

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()