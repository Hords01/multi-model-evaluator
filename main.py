import os
import pandas as pd
import numpy as np
import time
import torch
import torch.nn as nn
import torchvision.datasets as datasets
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision.models as models
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"using dev: {device}")

print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "no GPU available")

# creating results and cm folder
os.makedirs("models_results/training_history", exist_ok=True)
os.makedirs("models_results/confusion_matrices", exist_ok=True)

# transformers
train_transform = transforms.Compose([
    transforms.Resize((224, 224)), # for inceptionv3 use 299x299 / incepitonv3 input is only 299x299
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

# datasets, dataloaders
main_data_dir = "data/"
train_dataset = datasets.ImageFolder(root=os.path.join(main_data_dir, "train"), transform=train_transform)
val_dataset = datasets.ImageFolder(root=os.path.join(main_data_dir, "val"), transform=test_transform)
test_dataset = datasets.ImageFolder(root=os.path.join(main_data_dir, "test"), transform=test_transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)



model_dict = { # 27 models

    # *** alexnet -densenet
    #"AlexNet": models.alexnet(pretrained=False),
    #"DenseNet121": models.densenet121(pretrained=False),
    #"DenseNet161": models.densenet161(pretrained=False),
    #"DenseNet169": models.densenet169(pretrained=False),
    #"DenseNet201": models.densenet201(pretrained=False),
#
    ## *** efficientnet - googlenet - inceptionv3
    #"EfficientNet": models.efficientnet_b0(pretrained=False),
    #"GoogLeNet": models.googlenet(pretrained=False),
    #"InceptionV3": models.inception_v3(pretrained=False),
#
    ## *** mobilenet
    #"MobileNetV2": models.mobilenet_v2(pretrained=False),
    #"MobileNetV3_large": models.mobilenet_v3_large(pretrained=False),
    #"MobileNetV3_small": models.mobilenet_v3_small(pretrained=False),
#
    ## *** resnet
    #"ResNet18": models.resnet18(pretrained=False),
    #"ResNet34": models.resnet34(pretrained=False),
    #"ResNet50": models.resnet50(pretrained=False),
    #"ResNet101": models.resnet101(pretrained=False),
    #"ResNet152": models.resnet152(pretrained=False),
#
    ## *** resnet-wide
    #"ResNet50_wide": models.wide_resnet50_2(pretrained=False),
    #"ResNet101_wide": models.wide_resnet101_2(pretrained=False),
#
    ## *** resnext
    #"ResNeXt50_32x4d": models.resnext50_32x4d(pretrained=False),
    #"ResNeXt101_32x8d": models.resnext101_32x8d(pretrained=False),
    #"ResNeXt101_64x4d": models.resnext101_64x4d(pretrained=False),
#
#
    ## *** vgg
    #"VGG16": models.vgg16(pretrained=False),
    #"VGG16_bn": models.vgg16_bn(pretrained=False),
#
    ## *** shufflenet
    #"ShuffleNetV2_x0_5": models.shufflenet_v2_x0_5(pretrained=False),
    #"ShuffleNetV2_x1_0": models.shufflenet_v2_x1_0(pretrained=False),
    #"ShuffleNetV2_x1_5": models.shufflenet_v2_x1_5(pretrained=False),
    #"ShuffleNetV2_x2_0": models.shufflenet_v2_x2_0(pretrained=False),
}


def modify_model(model, num_classes=4):
    if hasattr(model, 'fc'):  # resnet, efficientnet
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif hasattr(model, 'classifier'):  # densenet, alexnet, vgg
        if isinstance(model.classifier, nn.Sequential):  # if sequential change last layer
            model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)
        else:  # if there is only one linear layer
            model.classifier = nn.Linear(model.classifier.in_features, num_classes)
    elif hasattr(model, 'aux_logits'):  # googlenet, incepitonv3 , for auxiliary classifiers
        model.aux_logits = True
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    else:
        raise ValueError(f"Unsupported model: {type(model).__name__}")

    return model


# - train
def train_model(model, model_name, epochs=10, lr=0.001):

    model = modify_model(model).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    total_params = sum(p.numel() for p in model.parameters()) # calculating params
    start_time = time.time() # initial time

    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

    for epoch in range(epochs):
        model.train()
        train_loss, correct, total = 0.0, 0, 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)

            if isinstance(model, models.GoogLeNet):  # auxiliary classifiers just for training
                main_output = outputs.logits
                aux_output = outputs.aux_logits1
                loss = criterion(main_output, labels) + 0.4 * criterion(aux_output, labels)

            elif isinstance(model, models.Inception3): # incepitonv3 auxiliary classifiers just for training
                outputs = model(images)
                main_output, aux_output = outputs[0], outputs[1]
                loss = criterion(main_output, labels) + 0.4 * criterion(aux_output, labels)

            else:
                loss = criterion(outputs if not isinstance(model, models.Inception3) else outputs[0], labels)


            loss.backward()
            optimizer.step()

            train_loss += loss.item()

            # !!!
            if isinstance(model, models.GoogLeNet):
                _, predicted = torch.max(outputs[0], 1)
            elif isinstance(model, models.Inception3):
                _, predicted = torch.max(outputs[0], 1)
            else:
                _, predicted = torch.max(outputs, 1)


            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        train_acc = 100 * correct / total
        history["train_loss"].append(train_loss / len(train_loader))
        history["train_acc"].append(train_acc)


        # - Val
        model.eval()
        val_loss, correct, total = 0.0, 0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)

                # !!!
                loss = criterion(outputs, labels) # ınceptionv3  main output
                val_loss += loss.item()
                _, predicted = torch.max(outputs, 1) # ınceptionv3  main output


                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        val_acc = 100 * correct / total
        history["val_loss"].append(val_loss / len(val_loader))
        history["val_acc"].append(val_acc)

        finish_time = time.time() # elapsed time for training

        print(
            f"{model_name} - Epoch {epoch + 1}: Train Loss={train_loss:.4f}, Train Acc={train_acc:.2f}%, Val Loss={val_loss:.4f}, Val Acc={val_acc:.2f}%")

    # Test acc / metrics
    test_acc, cm, report = evaluate_model(model, test_loader, model_name)
    torch.save(model.state_dict(), f"models_results/model_records/{model_name}.pt")
    return history, test_acc, finish_time, total_params, report



# - evaluate
def evaluate_model(model, test_loader, model_name):
    model.eval()
    correct, total = 0, 0
    all_preds, all_labels = [], []

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)

            # ı know it looks wierd, but ı couldn't fix it
            if isinstance(model, models.Inception3):
                _, predicted = torch.max(outputs, 1)
            elif isinstance(model, models.GoogLeNet):
                _, predicted = torch.max(outputs, 1)
            else:
                _, predicted = torch.max(outputs, 1)


            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    test_acc = 100 * correct / total
    cm = confusion_matrix(all_labels, all_preds)
    report = classification_report(all_labels, all_preds, output_dict=True) # metrics

    # Confusion Matrix
    plt.figure(figsize=(6, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=train_dataset.classes,
                yticklabels=train_dataset.classes)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(f"Confusion Matrix - {model_name}")
    plt.savefig(f"models_results/confusion_matrices/{model_name}.png")
    plt.close()

    return test_acc, cm, report



# Model training
results = []
for name, model in model_dict.items():
    history, test_acc, finish_time, total_params, report = train_model(model, name, epochs=10, lr=0.001)
    results.append({
        "Model": name,
        "Test Accuracy": test_acc,
        "Training Time (sec)": finish_time,
        "Total params": total_params,
        "Precision": report["weighted avg"]["precision"],
        "Recall": report["weighted avg"]["recall"],
        "F1": report["weighted avg"]["f1-score"],
    })
    df = pd.DataFrame(history)
    df.to_csv(f"models_results/training_history/{name}_history.csv", index=False)

# model comparison
pd.DataFrame(results).to_csv("models_results/model_comparison.csv", index=False)