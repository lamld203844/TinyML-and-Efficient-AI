import torch
import torch.nn as nn
from collections import defaultdict, OrderedDict

class VGG(nn.Module):
    # ARCH = [64, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M']
    # -> avgpool -> fc
    ARCH = [64, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M']
    
    def __init__(self, in_channels=3, out_features=10) -> None:
        super().__init__()
        
        layers = OrderedDict()
        count = defaultdict(int)
        
        def add(name, layer):
            """append [str, Module] to layers(OrderedDict) and update count"""
            layers[f"{name}{count[name]}"] = layer
            count[name] += 1
        
        for x in self.ARCH:
            if x != 'M':
                # append (Conv2d > Batchnorm > ReLU) layers
                add(f"conv", nn.Conv2d(in_channels, x, kernel_size=3, padding=1))
                add(f"bn", nn.BatchNorm2d(x))
                add(f"relu", nn.ReLU(inplace=True))
                in_channels = x
            else:
                # maxpool
                add(f"pool", nn.MaxPool2d(2))

        self.backbone = nn.Sequential(OrderedDict(layers))
        self.classifier = nn.Linear(in_channels, out_features)
    
    def forward(self, x):
        # input (N, 3, 32, 32) ---backbone---> (N, 512, 2, 2)
        x = self.backbone(x)
        
        # average pooling
        x = torch.mean(x, (2, 3))
        
        # feed-forward
        logits = self.classifier(x)
        
        return logits
    
    
if __name__ == "__main__":
    N=512 # batch size
    model = VGG()
    model.cpu()
    a = model(torch.rand(N, 3, 32, 32))
    assert a.shape == torch.Size([N, 10])