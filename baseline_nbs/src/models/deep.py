from torchvision.models.segmentation import deeplabv3_resnet50
import segmentation_models_pytorch as smp

def get_deep_model(model_name):
    if model_name == "deeplabv3":
        return deeplabv3_resnet50(weights=None, weights_backbone=None, num_classes=1)
    elif model_name == "resnet18":
        return smp.FPN(encoder_name="resnet18", encoder_weights=None, in_channels=3, classes=1)
    else:
        raise ValueError(f"Unknown deep model: {model_name}")