from torch import nn


def init_weights(m):
    """
    Applies the correct weight initialization to different layer types.
    THIS IS THE FIX.
    """
    if isinstance(m, (nn.Linear, nn.Conv2d)):
        # Kaiming (He) uniform initialization for ReLU
        # This is bounded and CANNOT create Inf
        nn.init.kaiming_uniform_(m.weight, mode='fan_out', nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)

    elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.GroupNorm)):
        # Initialize BatchNorm/GroupNorm weights to 1 and biases to 0
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)

    elif isinstance(m, nn.Embedding):
        # Initialize Embedding weights with a normal distribution
        nn.init.normal_(m.weight, mean=0.0, std=1.0)
        if m.padding_idx is not None:
            # Set the padding token embedding to all zeros
            nn.init.constant_(m.weight[m.padding_idx], 0)