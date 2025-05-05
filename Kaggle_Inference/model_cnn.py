import torchvision.models as models
import torch
import torch.nn as nn
def get_model(num_classes):
    """
    Crea un modelo ResNet18 modificado para aceptar 1 canal de entrada (en lugar de 3) 
    y ajusta la capa final para el número de clases especificado.
    """
    model = models.resnet18(pretrained=True)

    # Paso 2: Reemplazar la primera capa para aceptar 1 canal (no 3)
    model.conv1 = nn.Conv2d(
        in_channels=1, out_channels=64,
        kernel_size=7, stride=2, padding=3, bias=False
    )
    #Descomentar si se trabaja con librería OpenSoundScapes
    model.fc = nn.Linear(model.fc.in_features, num_classes)

    return model