import torchvision.models as models
import torch
import torch.nn as nn
import json

# ============================
# MODELO CNN PERSONALIZADO
# ============================
class CNNClassifier(nn.Module):
    def __init__(self, num_classes):
        super(CNNClassifier, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.dropout = nn.Dropout(0.4)
        self.fc1 = nn.Linear(128 * 4 * 12, 256)  # Ajustado para entrada (1,32,96)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(nn.functional.relu(self.fc1(x)))
        return self.fc2(x)

# ============================
# GET_MODEL
# ============================
def get_model(num_classes, model_name='resnet18', pretrained=True):
    """
    Crea un modelo a partir de torchvision o personalizado según el nombre.
    """

    if model_name == "cnn_custom":
        return CNNClassifier(num_classes)

    model_dict = {
        "alexnet": models.alexnet,
        "convnext_base": models.convnext_base,
        "convnext_large": models.convnext_large,
        "convnext_small": models.convnext_small,
        "convnext_tiny": models.convnext_tiny,
        "densenet121": models.densenet121,
        "densenet161": models.densenet161,
        "densenet169": models.densenet169,
        "densenet201": models.densenet201,
        "efficientnet_b0": models.efficientnet_b0,
        "efficientnet_b1": models.efficientnet_b1,
        "efficientnet_b2": models.efficientnet_b2,
        "efficientnet_b3": models.efficientnet_b3,
        "efficientnet_b4": models.efficientnet_b4,
        "efficientnet_b5": models.efficientnet_b5,
        "efficientnet_b6": models.efficientnet_b6,
        "efficientnet_b7": models.efficientnet_b7,
        "googlenet": models.googlenet,
        "inception_v3": models.inception_v3,
        "mnasnet0_5": models.mnasnet0_5,
        "mnasnet1_0": models.mnasnet1_0,
        "mobilenet_v2": models.mobilenet_v2,
        "mobilenet_v3_large": models.mobilenet_v3_large,
        "resnet18": models.resnet18,
        "resnet34": models.resnet34,
        "resnet50": models.resnet50,
        "resnet101": models.resnet101,
        "resnet152": models.resnet152,
        "resnext50_32x4d": models.resnext50_32x4d,
        "regnet_x_400mf": models.regnet_x_400mf,
        "shufflenet_v2_x1_0": models.shufflenet_v2_x1_0,
        "squeezenet1_0": models.squeezenet1_0,
        "swin_t": models.swin_t,
        "vit_b_16": models.vit_b_16,
        "wide_resnet50_2": models.wide_resnet50_2,
    }

    if model_name not in model_dict:
        raise ValueError(f"Modelo '{model_name}' no soportado.")

    model = model_dict[model_name](pretrained=pretrained)

    # Modificar la primera capa si es posible
    if hasattr(model, "conv1") and isinstance(model.conv1, nn.Conv2d):
        model.conv1 = nn.Conv2d(
            in_channels=1, out_channels=model.conv1.out_channels,
            kernel_size=model.conv1.kernel_size,
            stride=model.conv1.stride,
            padding=model.conv1.padding,
            bias=False
        )

    # Modificar la capa final si existe como 'fc'
    if hasattr(model, "fc") and isinstance(model.fc, nn.Linear):
        model.fc = nn.Linear(model.fc.in_features, num_classes)

    return model

# ============================
# MLP CONFIGURABLE
# ============================
class MLP(nn.Module):
    def __init__(self, input_size, hidden_sizes, num_classes, activation_fn=nn.ReLU):
        super(MLP, self).__init__()
        layers = [nn.Flatten()]
        layers.append(nn.Linear(input_size, hidden_sizes[0]))
        layers.append(activation_fn())
        for i in range(1, len(hidden_sizes)):
            layers.append(nn.Linear(hidden_sizes[i - 1], hidden_sizes[i]))
            layers.append(activation_fn())
        layers.append(nn.Linear(hidden_sizes[-1], num_classes))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.model(x)

def get_MLP(num_classes, input_size, hidden_sizes, activation_fn=nn.ReLU):
    return MLP(input_size, hidden_sizes, num_classes, activation_fn)
