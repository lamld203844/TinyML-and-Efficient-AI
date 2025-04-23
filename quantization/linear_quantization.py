import torch
from torch import nn


def get_quantization_scale_and_zero_point(fp_tensor, bitwidth):
    """
    get quantization scale for single tensor
    :param fp_tensor: [torch.(cuda.)Tensor] floating tensor to be quantized
    :param bitwidth: [int] quantization bit width
    :return:
        [float] scale
        [int] zero_point
    """
    quantized_min, quantized_max = get_quantized_range(bitwidth)
    fp_max = fp_tensor.max().item()
    fp_min = fp_tensor.min().item()

    ############### YOUR CODE STARTS HERE ###############
    # hint: one line of code for calculating scale
    scale = (fp_max - fp_min)/(quantized_max - quantized_min)
    # hint: one line of code for calculating zero_point
    zero_point = quantized_min - fp_min/scale
    ############### YOUR CODE ENDS HERE #################

    # clip the zero_point to fall in [quantized_min, quantized_max]
    if zero_point < quantized_min:
        zero_point = quantized_min
    elif zero_point > quantized_max:
        zero_point = quantized_max
    else: # convert from float to int using round()
        zero_point = round(zero_point)
    return scale, int(zero_point)

def get_quantization_scale_for_weight(weight, bitwidth):
    """
    get quantization scale for single tensor of weight
    :param weight: [torch.(cuda.)Tensor] floating weight to be quantized
    :param bitwidth: [integer] quantization bit width
    :return:
        [floating scalar] scale
    """
    # we just assume values in weight are symmetric
    # we also always make zero_point 0 for weight
    fp_max = max(weight.abs().max().item(), 5e-7)
    _, quantized_max = get_quantized_range(bitwidth)
    return fp_max / quantized_max

def linear_quantize_weight_per_channel(tensor, bitwidth):
    """
    linear quantization for weight tensor
        using different scales and zero_points for different output channels
    :param tensor: [torch.(cuda.)Tensor] floating weight to be quantized
    :param bitwidth: [int] quantization bit width
    :return:
        [torch.(cuda.)Tensor] quantized tensor
        [torch.(cuda.)Tensor] scale tensor
        [int] zero point (which is always 0)
    """
    dim_output_channels = 0
    num_output_channels = tensor.shape[dim_output_channels]
    scale = torch.zeros(num_output_channels, device=tensor.device)
    for oc in range(num_output_channels):
        _subtensor = tensor.select(dim_output_channels, oc)
        _scale = get_quantization_scale_for_weight(_subtensor, bitwidth)
        scale[oc] = _scale
    scale_shape = [1] * tensor.dim()
    scale_shape[dim_output_channels] = -1
    scale = scale.view(scale_shape)
    quantized_tensor = linear_quantize(tensor, bitwidth, scale, zero_point=0)
    return quantized_tensor, scale, 0

class QuantizedConv2d(nn.Module):
    def __init__(self, weight, bias,
                    input_zero_point, output_zero_point,
                    input_scale, weight_scale, output_scale,
                    stride, padding, dilation, groups,
                    feature_bitwidth=8, weight_bitwidth=8):
        super().__init__()
        # current version Pytorch does not support IntTensor as nn.Parameter
        self.register_buffer('weight', weight)
        self.register_buffer('bias', bias)

        self.input_zero_point = input_zero_point
        self.output_zero_point = output_zero_point

        self.input_scale = input_scale
        self.register_buffer('weight_scale', weight_scale)
        self.output_scale = output_scale

        self.stride = stride
        self.padding = (padding[1], padding[1], padding[0], padding[0])
        self.dilation = dilation
        self.groups = groups

        self.feature_bitwidth = feature_bitwidth
        self.weight_bitwidth = weight_bitwidth


    def forward(self, x):
        return quantized_conv2d(
            x, self.weight, self.bias,
            self.feature_bitwidth, self.weight_bitwidth,
            self.input_zero_point, self.output_zero_point,
            self.input_scale, self.weight_scale, self.output_scale,
            self.stride, self.padding, self.dilation, self.groups
            )

class QuantizedLinear(nn.Module):
    def __init__(self, weight, bias,
                 input_zero_point, output_zero_point,
                 input_scale, weight_scale, output_scale,
                 feature_bitwidth=8, weight_bitwidth=8):
        super().__init__()
        # current version Pytorch does not support IntTensor as nn.Parameter
        self.register_buffer('weight', weight)
        self.register_buffer('bias', bias)

        self.input_zero_point = input_zero_point
        self.output_zero_point = output_zero_point

        self.input_scale = input_scale
        self.register_buffer('weight_scale', weight_scale)
        self.output_scale = output_scale

        self.feature_bitwidth = feature_bitwidth
        self.weight_bitwidth = weight_bitwidth

    def forward(self, x):
        return quantized_linear(
            x, self.weight, self.bias,
            self.feature_bitwidth, self.weight_bitwidth,
            self.input_zero_point, self.output_zero_point,
            self.input_scale, self.weight_scale, self.output_scale
            )

class QuantizedMaxPool2d(nn.MaxPool2d):
    def forward(self, x):
        # current version PyTorch does not support integer-based MaxPool
        return super().forward(x.float()).to(torch.int8)

class QuantizedAvgPool2d(nn.AvgPool2d):
    def forward(self, x):
        # current version PyTorch does not support integer-based AvgPool
        return super().forward(x.float()).to(torch.int8)


if __name__ == "__main__":
    # --------------------------1.----------------------------------------
    # 1. fuse a BatchNorm layer into its previous convolutional layer
    def fuse_conv_bn(conv, bn):
        # modified from https://mmcv.readthedocs.io/en/latest/_modules/mmcv/cnn/utils/fuse_conv_bn.html
        assert conv.bias is None

        factor = bn.weight.data / torch.sqrt(bn.running_var.data + bn.eps)
        conv.weight.data = conv.weight.data * factor.reshape(-1, 1, 1, 1)
        conv.bias = nn.Parameter(- bn.running_mean.data * factor + bn.bias.data)

        return conv
    
    print('Before conv-bn fusion: backbone length', len(model.backbone))
    #  fuse the batchnorm into conv layers
    recover_model()
    model_fused = copy.deepcopy(model)
    fused_backbone = []
    ptr = 0
    while ptr < len(model_fused.backbone):
        if isinstance(model_fused.backbone[ptr], nn.Conv2d) and \
            isinstance(model_fused.backbone[ptr + 1], nn.BatchNorm2d):
            fused_backbone.append(fuse_conv_bn(
                model_fused.backbone[ptr], model_fused.backbone[ptr+ 1]))
            ptr += 2
        else:
            fused_backbone.append(model_fused.backbone[ptr])
            ptr += 1
    model_fused.backbone = nn.Sequential(*fused_backbone)

    print('After conv-bn fusion: backbone length', len(model_fused.backbone))
    # sanity check, no BN anymore
    for m in model_fused.modules():
        assert not isinstance(m, nn.BatchNorm2d)

    #  the accuracy will remain the same after fusion
    fused_acc = evaluate(model_fused, dataloader['test'])
    print(f'Accuracy of the fused model={fused_acc:.2f}%')
    # --------------------------1.----------------------------------------
        
    # --------------------------2. ----------------------------------------
    # 2. We will run the model with some sample data to get the range of each
    # feature map, so that we can get the range of the feature maps and compute
    # their corresponding scaling factors and zero points.
    # add hook to record the min max value of the activation
    input_activation = {}
    output_activation = {}

    def add_range_recoder_hook(model):
        import functools
        def _record_range(self, x, y, module_name):
            x = x[0]
            input_activation[module_name] = x.detach()
            output_activation[module_name] = y.detach()

        all_hooks = []
        for name, m in model.named_modules():
            if isinstance(m, (nn.Conv2d, nn.Linear, nn.ReLU)):
                all_hooks.append(m.register_forward_hook(
                    functools.partial(_record_range, module_name=name)))
        return all_hooks

    hooks = add_range_recoder_hook(model_fused)
    sample_data = iter(dataloader['train']).__next__()[0]
    model_fused(sample_data.cuda())

    # remove hooks
    for h in hooks:
        h.remove()
    # --------------------------2. ----------------------------------------
    
    # --------------------------3. ----------------------------------------
    # Model quantization: convert the model in the following mapping
    # 
    # nn.Conv2d: QuantizedConv2d,
    # nn.Linear: QuantizedLinear,
    # # the following twos are just wrappers, as current
    # # torch modules do not support int8 data format;
    # # we will temporarily convert them to fp32 for computation
    # nn.MaxPool2d: QuantizedMaxPool2d,
    # nn.AvgPool2d: QuantizedAvgPool2d,

        
    # we use int8 quantization, which is quite popular
    feature_bitwidth = weight_bitwidth = 8
    quantized_model = copy.deepcopy(model_fused)
    quantized_backbone = []
    ptr = 0
    while ptr < len(quantized_model.backbone):
        if isinstance(quantized_model.backbone[ptr], nn.Conv2d) and \
            isinstance(quantized_model.backbone[ptr + 1], nn.ReLU):
            conv = quantized_model.backbone[ptr]
            conv_name = f'backbone.{ptr}'
            relu = quantized_model.backbone[ptr + 1]
            relu_name = f'backbone.{ptr + 1}'

            input_scale, input_zero_point = \
                get_quantization_scale_and_zero_point(
                    input_activation[conv_name], feature_bitwidth)

            output_scale, output_zero_point = \
                get_quantization_scale_and_zero_point(
                    output_activation[relu_name], feature_bitwidth)

            quantized_weight, weight_scale, weight_zero_point = \
                linear_quantize_weight_per_channel(conv.weight.data, weight_bitwidth)
            quantized_bias, bias_scale, bias_zero_point = \
                linear_quantize_bias_per_output_channel(
                    conv.bias.data, weight_scale, input_scale)
            shifted_quantized_bias = \
                shift_quantized_conv2d_bias(quantized_bias, quantized_weight,
                                            input_zero_point)

            quantized_conv = QuantizedConv2d(
                quantized_weight, shifted_quantized_bias,
                input_zero_point, output_zero_point,
                input_scale, weight_scale, output_scale,
                conv.stride, conv.padding, conv.dilation, conv.groups,
                feature_bitwidth=feature_bitwidth, weight_bitwidth=weight_bitwidth
            )

            quantized_backbone.append(quantized_conv)
            ptr += 2
        elif isinstance(quantized_model.backbone[ptr], nn.MaxPool2d):
            quantized_backbone.append(QuantizedMaxPool2d(
                kernel_size=quantized_model.backbone[ptr].kernel_size,
                stride=quantized_model.backbone[ptr].stride
                ))
            ptr += 1
        elif isinstance(quantized_model.backbone[ptr], nn.AvgPool2d):
            quantized_backbone.append(QuantizedAvgPool2d(
                kernel_size=quantized_model.backbone[ptr].kernel_size,
                stride=quantized_model.backbone[ptr].stride
                ))
            ptr += 1
        else:
            raise NotImplementedError(type(quantized_model.backbone[ptr]))  # should not happen
    quantized_model.backbone = nn.Sequential(*quantized_backbone)

    # finally, quantized the classifier
    fc_name = 'classifier'
    fc = model.classifier
    input_scale, input_zero_point = \
        get_quantization_scale_and_zero_point(
            input_activation[fc_name], feature_bitwidth)

    output_scale, output_zero_point = \
        get_quantization_scale_and_zero_point(
            output_activation[fc_name], feature_bitwidth)

    quantized_weight, weight_scale, weight_zero_point = \
        linear_quantize_weight_per_channel(fc.weight.data, weight_bitwidth)
    quantized_bias, bias_scale, bias_zero_point = \
        linear_quantize_bias_per_output_channel(
            fc.bias.data, weight_scale, input_scale)
    shifted_quantized_bias = \
        shift_quantized_linear_bias(quantized_bias, quantized_weight,
                                    input_zero_point)

    quantized_model.classifier = QuantizedLinear(
        quantized_weight, shifted_quantized_bias,
        input_zero_point, output_zero_point,
        input_scale, weight_scale, output_scale,
        feature_bitwidth=feature_bitwidth, weight_bitwidth=weight_bitwidth
    )
    # --------------------------3. ----------------------------------------
