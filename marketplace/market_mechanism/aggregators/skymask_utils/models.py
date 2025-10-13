from collections import OrderedDict
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F

import marketplace.market_mechanism.aggregators.skymask_utils.mytorch as my


def create_masknet(param_list, net_type, ctx):
    """Create a mask network based on the architecture type."""

    nworker = len(param_list)

    if not param_list or not param_list[0]:
        raise ValueError("Empty parameter list provided to create_masknet")

    # Normalize net_type
    if net_type is None or net_type == 'None' or net_type == '':
        net_type = _infer_net_type_from_params(param_list)
        print(f"Auto-inferred net_type: {net_type}")

    net_type = net_type.lower()

    try:
        # Try to match specific architectures first
        if net_type == "cnn":
            if len(param_list[0]) == 8:
                print(f"✅ Using CNNMaskNet (exact match: 8 params)")
                masknet = CNNMaskNet(param_list, nworker, ctx)
            else:
                print(
                    f"⚠️  Warning: 'cnn' typically expects 8 params but got {len(param_list[0])}, using DynamicMaskNet")
                masknet = DynamicMaskNet(param_list, nworker, ctx)

        elif net_type in ["resnet20", "resnet18"]:
            print(f"✅ Using ResMaskNet")
            masknet = ResMaskNet(param_list, nworker, ctx)

        elif net_type == "lr":
            if len(param_list[0]) == 2:
                print(f"✅ Using LRMaskNet (exact match: 2 params)")
                masknet = LRMaskNet(param_list, nworker, ctx)
            else:
                print(f"⚠️  Warning: 'lr' expects 2 params but got {len(param_list[0])}, using DynamicMaskNet")
                masknet = DynamicMaskNet(param_list, nworker, ctx)

        elif net_type == "lenet":
            if len(param_list[0]) == 10:
                print(f"✅ Using LeNetMaskNet (exact match: 10 params)")
                masknet = LeNetMaskNet(param_list, nworker, ctx)
            else:
                print(f"⚠️  Warning: 'lenet' expects 10 params but got {len(param_list[0])}, using DynamicMaskNet")
                masknet = DynamicMaskNet(param_list, nworker, ctx)

        elif net_type == "cifarcnn":
            if len(param_list[0]) == 16:
                print(f"✅ Using CnnCifarMaskNet (exact match: 16 params)")
                masknet = CnnCifarMaskNet(param_list, nworker, ctx)
            else:
                print(f"⚠️  Warning: 'cifarcnn' expects 16 params but got {len(param_list[0])}, using DynamicMaskNet")
                masknet = DynamicMaskNet(param_list, nworker, ctx)

        elif net_type in ["flexiblecnn", "dynamic"]:
            print(f"✅ Using DynamicMaskNet (flexible architecture support)")
            masknet = DynamicMaskNet(param_list, nworker, ctx)

        else:
            # Unknown type - try dynamic as last resort
            print(f"⚠️  Warning: Unknown net_type '{net_type}', attempting DynamicMaskNet")
            masknet = DynamicMaskNet(param_list, nworker, ctx)

    except Exception as e:
        print(f"\n{'=' * 80}")
        print(f"ERROR: Failed to create masknet of type '{net_type}'")
        print(f"Reason: {e}")
        print(f"\nParameter structure:")
        print(f"  - Number of workers: {len(param_list)}")
        print(f"  - Params per worker: {len(param_list[0]) if param_list else 0}")
        if param_list and param_list[0]:
            print(f"  - Parameter shapes:")
            for i, p in enumerate(param_list[0][:30]):  # Show first 30
                param_type = "Conv2d" if len(p.shape) == 4 else "Linear" if len(p.shape) == 2 else f"{len(p.shape)}D"
                print(f"      Param {i:2d}: {str(p.shape):25s} ({p.numel():10d} elements) [{param_type}]")
        print(f"{'=' * 80}\n")
        raise RuntimeError(
            f"Cannot create masknet for type '{net_type}'. "
            f"Check the error above and parameter structure. "
            f"You may need to set sm_model_type='dynamic' in your config.")

    return masknet


def _infer_net_type_from_params(param_list):
    """Infer the network type from parameter structure."""
    if not param_list or not param_list[0]:
        return 'dynamic'

    num_params = len(param_list[0])

    # Count parameter types
    conv_count = sum(1 for p in param_list[0] if len(p.shape) == 4)
    linear_count = sum(1 for p in param_list[0] if len(p.shape) == 2)

    print(f"Parameter analysis: total={num_params}, conv_layers={conv_count}, linear_layers={linear_count}")

    # Exact matches for known structures ONLY if param count matches exactly
    if num_params == 2 and linear_count == 1:
        return 'lr'
    elif num_params == 8 and conv_count == 2:
        return 'cnn'
    elif num_params == 10 and conv_count == 2:
        return 'lenet'
    elif num_params == 16 and conv_count == 3:
        return 'cifarcnn'
    elif num_params > 100:
        return 'resnet18'
    else:
        # Use dynamic for anything else - SAFER DEFAULT
        print(f"No exact match found (params={num_params}), using 'dynamic'")
        return 'dynamic'


class DynamicMaskNet(nn.Module):
    """
    Dynamically builds a mask network by analyzing the parameter structure.
    Supports arbitrary CNN architectures with conv, batchnorm, and linear layers.
    """

    def __init__(self, worker_param_list: List[List[torch.Tensor]], num_workers: int, device: torch.device):
        super(DynamicMaskNet, self).__init__()

        self.num_workers = num_workers
        self.device = device

        if not worker_param_list or not worker_param_list[0]:
            raise ValueError("Empty parameter list")

        num_params = len(worker_param_list[0])
        print(f"\n{'=' * 80}")
        print(f"Building DynamicMaskNet from {num_params} parameters")
        print(f"{'=' * 80}")

        self.conv_layers = nn.ModuleList()
        self.bn_layers = nn.ModuleList()
        self.fc_layers = nn.ModuleList()

        # Parse parameters and build layers
        i = 0
        layer_idx = 0
        self.layer_sequence = []  # Track the order of operations

        while i < num_params:
            param = worker_param_list[0][i]

            # Conv2d layer (4D weight)
            if len(param.shape) == 4:
                out_channels = param.shape[0]

                # Check for bias
                has_bias = False
                bias_param = None
                if i + 1 < num_params:
                    next_param = worker_param_list[0][i + 1]
                    if len(next_param.shape) == 1 and next_param.shape[0] == out_channels:
                        has_bias = True
                        bias_param = next_param

                if has_bias:
                    print(f"  [{layer_idx}] Conv2d: weight={param.shape}, bias={bias_param.shape}")
                    conv_layer = my.myconv2d(
                        num_workers, device,
                        [w[i] for w in worker_param_list],
                        [w[i + 1] for w in worker_param_list],
                        padding=1
                    )
                    i += 2
                else:
                    print(f"  [{layer_idx}] Conv2d: weight={param.shape}, no bias")
                    conv_layer = my.myconv2d(
                        num_workers, device,
                        [w[i] for w in worker_param_list],
                        padding=1
                    )
                    i += 1

                self.conv_layers.append(conv_layer)
                self.layer_sequence.append(('conv', len(self.conv_layers) - 1))

                # Check for BatchNorm after conv
                if i + 1 < num_params:
                    p1 = worker_param_list[0][i]
                    p2 = worker_param_list[0][i + 1]

                    # BatchNorm has weight and bias of same size as conv output channels
                    if (len(p1.shape) == 1 and len(p2.shape) == 1 and
                            p1.shape[0] == out_channels and p2.shape[0] == out_channels):
                        print(f"  [{layer_idx}] BatchNorm: weight={p1.shape}, bias={p2.shape}")
                        bn_layer = my.mybatch_norm(
                            num_workers, device,
                            [w[i] for w in worker_param_list],
                            [w[i + 1] for w in worker_param_list]
                        )
                        self.bn_layers.append(bn_layer)
                        self.layer_sequence.append(('bn', len(self.bn_layers) - 1))
                        i += 2

                # Add activation and pooling after each conv block
                self.layer_sequence.append(('relu', None))
                self.layer_sequence.append(('pool', None))
                layer_idx += 1

            # Linear layer (2D weight)
            elif len(param.shape) == 2:
                in_features = param.shape[1]
                out_features = param.shape[0]

                # Check for bias
                has_bias = False
                if i + 1 < num_params:
                    next_param = worker_param_list[0][i + 1]
                    if len(next_param.shape) == 1 and next_param.shape[0] == out_features:
                        has_bias = True

                if has_bias:
                    print(
                        f"  [{layer_idx}] Linear: weight={param.shape} (in={in_features}, out={out_features}), bias={worker_param_list[0][i + 1].shape}")
                    fc_layer = my.mylinear(
                        num_workers, device,
                        [w[i] for w in worker_param_list],
                        [w[i + 1] for w in worker_param_list]
                    )
                    i += 2
                else:
                    print(
                        f"  [{layer_idx}] Linear: weight={param.shape} (in={in_features}, out={out_features}), no bias")
                    fc_layer = my.mylinear(
                        num_workers, device,
                        [w[i] for w in worker_param_list]
                    )
                    i += 1

                self.fc_layers.append(fc_layer)
                self.layer_sequence.append(('fc', len(self.fc_layers) - 1))

                # Add ReLU for all but last FC layer
                # Check if this is the last linear layer (no more 2D params ahead)
                is_last_fc = True
                for j in range(i, num_params):
                    if len(worker_param_list[0][j].shape) == 2:
                        is_last_fc = False
                        break

                if not is_last_fc:
                    self.layer_sequence.append(('relu', None))

                layer_idx += 1

            else:
                # Skip unknown parameter types (e.g., running_mean, running_var from BatchNorm)
                print(f"  [SKIP] Param {i}: shape={param.shape}")
                i += 1

        print(f"{'=' * 80}")
        print(f"Built network with:")
        print(f"  - {len(self.conv_layers)} conv layers")
        print(f"  - {len(self.bn_layers)} batchnorm layers")
        print(f"  - {len(self.fc_layers)} fc layers")
        print(f"  - Layer sequence length: {len(self.layer_sequence)}")
        print(f"{'=' * 80}\n")

        self.pool = nn.MaxPool2d(2, 2)

    def forward(self, x):
        need_flatten = True

        for op_type, idx in self.layer_sequence:
            if op_type == 'conv':
                x = self.conv_layers[idx](x)
            elif op_type == 'bn':
                x = self.bn_layers[idx](x)
            elif op_type == 'relu':
                x = F.relu(x)
            elif op_type == 'pool':
                x = self.pool(x)
            elif op_type == 'fc':
                if need_flatten:
                    x = torch.flatten(x, start_dim=1)
                    need_flatten = False
                x = self.fc_layers[idx](x)

        x = F.log_softmax(x, dim=1)
        return x


class CNNMaskNet(nn.Module):
    def __init__(self, param_list, num_workers, device):
        super(CNNMaskNet, self).__init__()

        self.num_workers = num_workers
        self.param_list = param_list
        self.device = device

        self.conv1 = nn.Sequential(OrderedDict([
            ('conv1', my.myconv2d(num_workers, device, [x[0] for x in param_list], [x[1] for x in param_list])),
            ('relu1', nn.ReLU()),
            ('maxpool1', nn.MaxPool2d(2, 2))])
        )

        self.conv2 = nn.Sequential(OrderedDict([
            ('conv1', my.myconv2d(num_workers, device, [x[2] for x in param_list], [x[3] for x in param_list])),
            ('relu1', nn.ReLU()),
            ('maxpool1', nn.MaxPool2d(2, 2))])
        )

        self.fc1 = my.mylinear(num_workers, device, [x[4] for x in param_list], [x[5] for x in param_list])
        self.fc2 = my.mylinear(num_workers, device, [x[6] for x in param_list], [x[7] for x in param_list])

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = torch.flatten(x, start_dim=1, end_dim=-1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        x = F.log_softmax(x, dim=1)  # Changed from dim=0 to dim=1 (correct for batch)
        return x


class ResidualBlock(nn.Module):
    """A standard ResNet residual block."""

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1, use_batch_norm: bool = True):
        super(ResidualBlock, self).__init__()

        # Main path
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels) if use_batch_norm else nn.Identity()
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels) if use_batch_norm else nn.Identity()

        # Shortcut connection to match dimensions
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels) if use_batch_norm else nn.Identity()
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResidualMaskBlock(nn.Module):
    def __init__(self, param_list, num_workers, device, stride=1):
        super(ResidualMaskBlock, self).__init__()

        self.num_workers = num_workers
        self.param_list = param_list
        self.device = device

        self.left = nn.Sequential(
            my.myconv2d(num_workers, device, [x[0] for x in param_list], stride=stride, padding=1),
            my.mybatch_norm(num_workers, device, [x[1] for x in param_list], [x[2] for x in param_list]),
            nn.ReLU(),
            my.myconv2d(num_workers, device, [x[5] for x in param_list], padding=1),
            my.mybatch_norm(num_workers, device, [x[6] for x in param_list], [x[7] for x in param_list])
        )
        self.shortcut = nn.Sequential()
        if len(param_list[0]) > 10:
            self.shortcut = nn.Sequential(
                my.myconv2d(num_workers, device, [x[10] for x in param_list], stride=stride),
                my.mybatch_norm(num_workers, device, [x[11] for x in param_list], [x[12] for x in param_list])
            )

    def forward(self, x):
        out = self.left(x)
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResMaskNet(nn.Module):
    def __init__(self, param_list, num_workers, device):
        super(ResMaskNet, self).__init__()

        self.num_workers = num_workers
        self.param_list = param_list
        self.device = device
        self.conv1 = nn.Sequential(
            my.myconv2d(num_workers, device, [x[0] for x in param_list], stride=1, padding=1),
            my.mybatch_norm(num_workers, device, [x[1] for x in param_list], [x[2] for x in param_list]),
            nn.ReLU()
        )
        self.layer1 = self.make_layer(ResidualMaskBlock, [x[5:35] for x in param_list], num_workers, device, 3,
                                      stride=1)  # 10+10+10
        self.layer2 = self.make_layer(ResidualMaskBlock, [x[35:70] for x in param_list], num_workers, device, 3,
                                      stride=2)  # 15+10+10
        self.layer3 = self.make_layer(ResidualMaskBlock, [x[70:105] for x in param_list], num_workers, device, 3,
                                      stride=2)  # 15+10+10
        self.fc = my.mylinear(num_workers, device, [x[105] for x in param_list], [x[106] for x in param_list])

    def make_layer(self, block, param_list, num_workers, device, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        head = 0
        for stride in strides:
            if stride == 1:
                layers.append(block([x[head:head + 10] for x in param_list], num_workers, device, stride))
                head += 10
            else:
                layers.append(block([x[head:head + 15] for x in param_list], num_workers, device, stride))
                head += 15

        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.avg_pool2d(out, 8)
        out = out.view(out.size(0), -1)
        out = self.fc(out)

        out = F.log_softmax(out, dim=0)
        return out


class LR(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(561, 6)

    def forward(self, x):
        y_pred = self.linear(x)
        out = F.log_softmax(y_pred, dim=0)
        return out


class LRMaskNet(nn.Module):
    def __init__(self, param_list, num_workers, device):
        super(LRMaskNet, self).__init__()

        self.num_workers = num_workers
        self.param_list = param_list
        self.device = device

        self.fc = my.mylinear(num_workers, device, [x[0] for x in param_list], [x[1] for x in param_list])

    def forward(self, x):
        y_pred = self.fc(x)
        out = F.log_softmax(y_pred, dim=0)
        return out


class LeNetMaskNet(nn.Module):
    """MaskNet specifically mirroring the LeNet architecture."""

    def __init__(self, worker_param_list: List[List[torch.Tensor]], num_workers: int, device: torch.device):
        """
        Args:
            worker_param_list: List (size=num_workers) where each element is a list
                               of parameters for one worker's LeNet model IN ORDER:
                               [conv1.w, conv1.b, conv2.w, conv2.b, fc1.w, fc1.b,
                                fc2.w, fc2.b, fc3.w, fc3.b]
            num_workers: Number of workers.
            device: Torch device.
        """
        super(LeNetMaskNet, self).__init__()
        self.num_workers = num_workers
        self.device = device

        # --- Parameter Validation ---
        if len(worker_param_list) != num_workers:
            raise ValueError(f"Expected {num_workers} lists in worker_param_list, got {len(worker_param_list)}")
        expected_params = 10
        if any(len(p) != expected_params for p in worker_param_list):
            raise ValueError(
                f"Each inner list in worker_param_list must have exactly {expected_params} tensors for LeNet.")

        # --- Layer Definitions (using my.* layers and hardcoded indices) ---
        # Layer params are passed as lists extracted from worker_param_list
        self.conv1 = my.myconv2d(num_workers, device,
                                 [w[0] for w in worker_param_list],
                                 [w[1] for w in worker_param_list])  # Removed kernel_size
        # Add stride, padding etc. IF they are non-default and needed by F.conv2d
        self.conv2 = my.myconv2d(num_workers, device,
                                 [w[2] for w in worker_param_list],
                                 [w[3] for w in worker_param_list])  # Removed kernel_size
        # Add stride, padding etc. IF non-default
        # Note: Input size for fc1 depends on image size (16*4*4 for 28x28 MNIST)
        self.fc1 = my.mylinear(num_workers, device,
                               [w[4] for w in worker_param_list],
                               [w[5] for w in worker_param_list])
        self.fc2 = my.mylinear(num_workers, device,
                               [w[6] for w in worker_param_list],
                               [w[7] for w in worker_param_list])
        self.fc3 = my.mylinear(num_workers, device,
                               [w[8] for w in worker_param_list],
                               [w[9] for w in worker_param_list])

    def forward(self, x):
        # --- Mimic LeNet forward pass ---
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(x.size()[0], -1)  # Flatten
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        # Optional: Apply LogSoftmax if MaskNet is used for classification during training
        # x = F.log_softmax(x, dim=1)
        return x

    def update(self, worker_param_list: List[List[torch.Tensor]]):
        """Updates the internal lists of worker parameters."""
        if len(worker_param_list) != self.num_workers or any(len(p) != 10 for p in worker_param_list):
            raise ValueError("Invalid parameter list structure for update.")
        # Call update on each my.* layer
        self.conv1.update([w[0] for w in worker_param_list], [w[1] for w in worker_param_list])
        self.conv2.update([w[2] for w in worker_param_list], [w[3] for w in worker_param_list])
        self.fc1.update([w[4] for w in worker_param_list], [w[5] for w in worker_param_list])
        self.fc2.update([w[6] for w in worker_param_list], [w[7] for w in worker_param_list])
        self.fc3.update([w[8] for w in worker_param_list], [w[9] for w in worker_param_list])


class CnnCifarMaskNet(nn.Module):
    """MaskNet specifically mirroring the CNN_CIFAR architecture."""

    def __init__(self, worker_param_list: List[List[torch.Tensor]], num_workers: int, device: torch.device,
                 num_classes=10):
        """
        Args:
            worker_param_list: List (size=num_workers) where each element is a list
                               of parameters for one worker's CNN_CIFAR model IN ORDER:
                               [conv1.w, conv1.b, bn1.w, bn1.b, conv2.w, conv2.b, bn2.w, bn2.b,
                                conv3.w, conv3.b, bn3.w, bn3.b, fc1.w, fc1.b, fc2.w, fc2.b]
            num_workers: Number of workers.
            device: Torch device.
            num_classes: Number of output classes for fc2.
        """
        super(CnnCifarMaskNet, self).__init__()
        self.num_workers = num_workers
        self.device = device

        # --- Parameter Validation ---
        if len(worker_param_list) != num_workers:
            raise ValueError(f"Expected {num_workers} lists in worker_param_list, got {len(worker_param_list)}")
        expected_params = 16
        if any(len(p) != expected_params for p in worker_param_list):
            raise ValueError(
                f"Each inner list in worker_param_list must have exactly {expected_params} tensors for CNN_CIFAR.")

        # --- Layer Definitions ---
        self.conv1 = my.myconv2d(num_workers, device, [w[0] for w in worker_param_list],
                                 [w[1] for w in worker_param_list], padding=1)
        self.bn1 = my.mybatch_norm(num_workers, device, [w[2] for w in worker_param_list],
                                   [w[3] for w in worker_param_list])
        self.conv2 = my.myconv2d(num_workers, device, [w[4] for w in worker_param_list],
                                 [w[5] for w in worker_param_list], padding=1)
        self.bn2 = my.mybatch_norm(num_workers, device, [w[6] for w in worker_param_list],
                                   [w[7] for w in worker_param_list])
        self.conv3 = my.myconv2d(num_workers, device, [w[8] for w in worker_param_list],
                                 [w[9] for w in worker_param_list], padding=1)
        self.bn3 = my.mybatch_norm(num_workers, device, [w[10] for w in worker_param_list],
                                   [w[11] for w in worker_param_list])
        self.pool = nn.MaxPool2d(2, 2)  # Standard layer
        # Note: Input features for fc1 depends on pooling (256 * 4 * 4 for CIFAR 32->16->8->4)
        self.fc1 = my.mylinear(num_workers, device, [w[12] for w in worker_param_list],
                               [w[13] for w in worker_param_list])
        self.fc2 = my.mylinear(num_workers, device, [w[14] for w in worker_param_list],
                               [w[15] for w in worker_param_list])

    def forward(self, x):
        # --- Mimic CNN_CIFAR forward pass ---
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool(x)
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        # Optional: Apply LogSoftmax if MaskNet is used for classification during training
        # x = F.log_softmax(x, dim=1)
        return x

    def update(self, worker_param_list: List[List[torch.Tensor]]):
        """Updates the internal lists of worker parameters."""
        if len(worker_param_list) != self.num_workers or any(len(p) != 16 for p in worker_param_list):
            raise ValueError("Invalid parameter list structure for update.")
        # Call update on each my.* layer
        self.conv1.update([w[0] for w in worker_param_list], [w[1] for w in worker_param_list])
        self.bn1.update([w[2] for w in worker_param_list], [w[3] for w in worker_param_list])
        self.conv2.update([w[4] for w in worker_param_list], [w[5] for w in worker_param_list])
        self.bn2.update([w[6] for w in worker_param_list], [w[7] for w in worker_param_list])
        self.conv3.update([w[8] for w in worker_param_list], [w[9] for w in worker_param_list])
        self.bn3.update([w[10] for w in worker_param_list], [w[11] for w in worker_param_list])
        self.fc1.update([w[12] for w in worker_param_list], [w[13] for w in worker_param_list])
        self.fc2.update([w[14] for w in worker_param_list], [w[15] for w in worker_param_list])


class FlexibleCNNMaskNet(nn.Module):
    """
    Generic MaskNet that dynamically adapts to the parameter structure.
    Works with ConfigurableFlexibleCNN and similar architectures.
    """

    def __init__(self, worker_param_list: List[List[torch.Tensor]], num_workers: int, device: torch.device):
        super(FlexibleCNNMaskNet, self).__init__()

        self.num_workers = num_workers
        self.device = device
        self.param_list = worker_param_list

        if not worker_param_list or not worker_param_list[0]:
            raise ValueError("Empty parameter list provided")

        # Analyze parameter structure
        self.layers = nn.ModuleList()
        num_params = len(worker_param_list[0])

        print(f"Building FlexibleCNNMaskNet with {num_params} parameters")

        i = 0
        while i < num_params:
            param = worker_param_list[0][i]

            if len(param.shape) == 4:  # Conv2d weight
                # Check if next param is bias (1D) or BatchNorm (also 1D but different size)
                if i + 1 < num_params:
                    next_param = worker_param_list[0][i + 1]

                    if len(next_param.shape) == 1 and next_param.shape[0] == param.shape[0]:
                        # This is a conv bias
                        print(f"  Layer {len(self.layers)}: Conv2d (params {i}, {i + 1})")
                        self.layers.append(
                            my.myconv2d(num_workers, device,
                                        [w[i] for w in worker_param_list],
                                        [w[i + 1] for w in worker_param_list],
                                        padding=1)
                        )
                        i += 2

                        # Check for BatchNorm after conv
                        if i + 1 < num_params and len(worker_param_list[0][i].shape) == 1:
                            print(f"  Layer {len(self.layers)}: BatchNorm (params {i}, {i + 1})")
                            self.layers.append(
                                my.mybatch_norm(num_workers, device,
                                                [w[i] for w in worker_param_list],
                                                [w[i + 1] for w in worker_param_list])
                            )
                            i += 2

                        self.layers.append(nn.ReLU())
                        self.layers.append(nn.MaxPool2d(2, 2))
                    else:
                        # Conv without bias (shouldn't happen but handle it)
                        print(f"  Layer {len(self.layers)}: Conv2d without bias (param {i})")
                        i += 1
                else:
                    i += 1

            elif len(param.shape) == 2:  # Linear weight
                # Check if next is bias
                if i + 1 < num_params and len(worker_param_list[0][i + 1].shape) == 1:
                    print(f"  Layer {len(self.layers)}: Linear (params {i}, {i + 1})")
                    self.layers.append(
                        my.mylinear(num_workers, device,
                                    [w[i] for w in worker_param_list],
                                    [w[i + 1] for w in worker_param_list])
                    )
                    i += 2

                    # Add ReLU except for last layer
                    if i < num_params:
                        self.layers.append(nn.ReLU())
                else:
                    print(f"  Layer {len(self.layers)}: Linear without bias (param {i})")
                    i += 1
            else:
                # Skip unknown parameter types
                print(f"  Skipping param {i} with shape {param.shape}")
                i += 1

        print(f"FlexibleCNNMaskNet built with {len(self.layers)} layers")

    def forward(self, x):
        # Track when to flatten
        need_flatten = True

        for i, layer in enumerate(self.layers):
            if isinstance(layer, (my.mylinear, nn.Linear)) and need_flatten:
                x = torch.flatten(x, start_dim=1)
                need_flatten = False

            x = layer(x)

        # Apply log_softmax on final output
        x = F.log_softmax(x, dim=1)
        return x
