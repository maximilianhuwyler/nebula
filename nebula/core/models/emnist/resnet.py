import matplotlib
import matplotlib.pyplot as plt
from torch import nn
from torchmetrics import MetricCollection

from nebula.core.models.nebulamodel import NebulaModel

matplotlib.use("Agg")
plt.switch_backend("Agg")
import torch
from torchmetrics.classification import (
    MulticlassAccuracy,
    MulticlassConfusionMatrix,
    MulticlassF1Score,
    MulticlassPrecision,
    MulticlassRecall,
)
from torchvision.models import resnet18, resnet34, resnet50

IMAGE_SIZE = 28

BATCH_SIZE = 256 if torch.cuda.is_available() else 64

classifiers = {
    "resnet18": resnet18(),
    "resnet34": resnet34(),
    "resnet50": resnet50(),
}


class EMNISTModelResNet(NebulaModel):
    def __init__(
        self,
        input_channels=1,  # Changed default to 1 for MNIST
        num_classes=10,
        learning_rate=1e-3,
        metrics=None,
        confusion_matrix=None,
        seed=None,
        implementation="scratch",
        classifier="resnet9",
    ):
        super().__init__()
        if metrics is None:
            metrics = MetricCollection([
                MulticlassAccuracy(num_classes=num_classes),
                MulticlassPrecision(num_classes=num_classes),
                MulticlassRecall(num_classes=num_classes),
                MulticlassF1Score(num_classes=num_classes),
            ])
        self.train_metrics = metrics.clone(prefix="Train/")
        self.val_metrics = metrics.clone(prefix="Validation/")
        self.test_metrics = metrics.clone(prefix="Test/")

        if confusion_matrix is None:
            self.cm = MulticlassConfusionMatrix(num_classes=num_classes)
        if seed is not None:
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)

        self.implementation = implementation
        self.classifier = classifier

        self.example_input_array = torch.rand(1, 1, 28, 28)  # Updated for MNIST dimensions
        self.learning_rate = learning_rate

        self.criterion = torch.nn.CrossEntropyLoss()

        self.model = self._build_model(input_channels, num_classes)

        self.epoch_global_number = {"Train": 0, "Validation": 0, "Test": 0}

    def _build_model(self, input_channels, num_classes):
        if self.implementation == "scratch":
            if self.classifier == "resnet9":
                def conv_block(input_channels, num_classes, pool=False):
                    # Reduce initial channel sizes for MNIST's simpler features
                    layers = [
                        nn.Conv2d(input_channels, num_classes, kernel_size=3, padding=1),
                        nn.BatchNorm2d(num_classes),
                        nn.ReLU(inplace=True),
                    ]
                    if pool:
                        # Use smaller 2x2 pooling to preserve features
                        layers.append(nn.MaxPool2d(2))
                    return nn.Sequential(*layers)

                # Reduce initial channel complexity
                conv1 = conv_block(input_channels, 32)  # Changed from 64
                conv2 = conv_block(32, 64, pool=True)   # Changed from 128
                res1 = nn.Sequential(conv_block(64, 64), conv_block(64, 64))

                conv3 = conv_block(64, 128, pool=True)  # Changed from 256
                conv4 = conv_block(128, 256, pool=True) # Changed from 512
                res2 = nn.Sequential(conv_block(256, 256), conv_block(256, 256))

                # Adjust final pooling size for MNIST
                classifier = nn.Sequential(nn.MaxPool2d(3), nn.Flatten(), nn.Linear(256, num_classes))

                return nn.ModuleDict({
                    "conv1": conv1,
                    "conv2": conv2,
                    "res1": res1,
                    "conv3": conv3,
                    "conv4": conv4,
                    "res2": res2,
                    "classifier": classifier,
                })

            if self.implementation in classifiers:
                model = classifiers[self.classifier]
                model.fc = torch.nn.Linear(model.fc.in_features, self.num_classes)
                return model

            raise NotImplementedError()

        if self.implementation == "timm":
            raise NotImplementedError()

        raise NotImplementedError()

    def forward(self, x):
        if not isinstance(x, torch.Tensor):
            raise TypeError(f"images must be a torch.Tensor, got {type(x)}")

        if self.implementation == "scratch":
            if self.classifier == "resnet9":
                out = self.model["conv1"](x)
                out = self.model["conv2"](out)
                out = self.model["res1"](out) + out
                out = self.model["conv3"](out)
                out = self.model["conv4"](out)
                out = self.model["res2"](out) + out
                out = self.model["classifier"](out)
                return out

            return self.model(x)
        if self.implementation == "timm":
            raise NotImplementedError()

        raise NotImplementedError()

    def configure_optimizers(self):
        if self.implementation == "scratch" and self.classifier == "resnet9":
            params = []
            for key, module in self.model.items():
                params += list(module.parameters())
            optimizer = torch.optim.Adam(params, lr=self.learning_rate, weight_decay=1e-4)
        else:
            optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate, weight_decay=1e-4)
        return optimizer
