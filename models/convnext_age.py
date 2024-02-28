import cv2
import timm
import torch
import torch.nn as nn


class RegressionModel(nn.Module):
    """Wrapper class for TIMM models."""

    def __init__(
        self,
        model_name: str,
        pretrained: bool = True,
        exportable: bool = True,
        global_pool: str = "avg",
        num_classes: int = 1,
    ) -> None:
        """
        Initialize the model from the timm repository.

        Args:
            model_name: name of timm model;
            pretrained: flag whether or not use
                pretrained ImageNet model;
            exportable: flag whether or not create
                exportable to ONNX model;
            global_pool: type of output global pooling;
            num_classes: number of output classes.
        """
        super().__init__()
        self.num_classes = num_classes
        self._backbone = timm.create_model(
            model_name=model_name,
            pretrained=pretrained,
            exportable=exportable,
            num_classes=self.num_classes,
            global_pool=global_pool,
        )

        in_features = self._backbone.get_classifier().in_features
        self._backbone.reset_classifier(num_classes=0)
        self.fc = torch.nn.Sequential(
            torch.nn.BatchNorm1d(num_features=in_features),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(in_features=in_features, out_features=self.num_classes),
        )

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        """Perform forward pass over input tensors.

        Args:
            tensor: Batch of images with shape (N, C, H, W).

        Returns:
            Raw logits with shape (N, 1).
        """
        return self.fc(self._backbone(tensor))
