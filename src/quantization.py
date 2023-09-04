
import typing as tp
import torch
from torch.utils.data import Dataloader
from torchvision.models.detection.image_list import ImageList
from torchvision.models.quantization import resnet
from torchvision.models.detection import backbone_utils
from torchvision.ops import feature_pyramid_network
from torch.ao.nn.quantized import FloatFunctional
import torch.nn.functional as F
from collections import OrderedDict


def normalize(images: torch.Tensor,
              image_mean: tp.List[float] = [0.485, 0.456, 0.406],
              image_std: tp.List[float] = [0.229, 0.224, 0.225]) -> torch.Tensor:
    """
    This is supposed to replace the normalization performed for FasterRCNN torchvision.

    Args:
        images (torch.Tensor): Images to be normalized. Shape: [N, C, H, W]
        image_mean (tp.List[float], optional): Mean values oer channel. Defaults to [0.485, 0.456, 0.406].
        image_std (tp.List[float], optional): _description_. Defaults to [0.229, 0.224, 0.225].

    Returns:
        torch.Tensor: Normalized Images. Shape: [N, C, H, W]
    """
    mean = torch.as_tensor(image_mean)
    std = torch.as_tensor(image_std)
    return (images - mean[None, :, None, None]) / std[None, :, None, None]


class FeaturePyramidNetwork(feature_pyramid_network.FeaturePyramidNetwork):
    """
    Reimplementation of `torchvision.ops.feature_pyramid_network.FeaturePyramidNetwork` to support
    stateful ops(to assist quantization of FPN) and dequant stubs that are required for dict outputs.
    """
    def __init__(
        self,
        in_channels_list: tp.List[int],
        out_channels: int,
        extra_blocks: tp.Optional[feature_pyramid_network.ExtraFPNBlock] = None,
        norm_layer: tp.Optional[tp.Callable[..., torch.nn.Module]] = None,
        return_layers: tp.Optional[tp.Dict[str, str]] = None
    ):

        super().__init__(in_channels_list, out_channels, extra_blocks, norm_layer)

        print(return_layers.keys())
        self.add_inner_list = torch.nn.ModuleList()

        for idx in range(len(list(return_layers.keys())) - 2, -1, -1):
            self.add_inner_list.append(FloatFunctional())


        # TODO: Remove this, DeQuantStub is not stateful, so only one instance is enough.
        self.dequants = torch.nn.ModuleList()
        for idx in range(len(list(return_layers.keys()))):
            self.dequants.append(torch.quantization.DeQuantStub())

        if extra_blocks is not None:
            if type(extra_blocks).__name__ == 'LastLevelMaxPool':
                self.dequants.append(torch.quantization.DeQuantStub())
            elif type(extra_blocks).__name__ == 'LastLevelP6P7':
                self.dequants.extend([torch.quantization.DeQuantStub(), torch.quantization.DeQuantStub()])


    def forward(self, x: tp.Dict[str, torch.Tensor]) -> tp.Dict[str, torch.Tensor]:
        """
        Computes the FPN for a set of feature maps.

        Updates:
            Replace add with torch functional ops. and  add dequant to dictionary.

        Args:
            x (OrderedDict[Tensor]): feature maps for each feature level.

        Returns:
            results (OrderedDict[Tensor]): feature maps after FPN layers.
                They are ordered from highest resolution first.
        """
        # unpack OrderedDict into two lists for easier handling
        names = list(x.keys())
        x = list(x.values())

        last_inner = self.get_result_from_inner_blocks(x[-1], -1)
        results = []
        results.append(self.get_result_from_layer_blocks(last_inner, -1))

        for idx in range(len(x) - 2, -1, -1):
            inner_lateral = self.get_result_from_inner_blocks(x[idx], idx)
            feat_shape = inner_lateral.shape[-2:]
            inner_top_down = F.interpolate(last_inner, size=feat_shape, mode="nearest")
            last_inner = self.add_inner_list[idx].add(inner_lateral, inner_top_down)
            results.insert(0, self.get_result_from_layer_blocks(last_inner, idx))

        if self.extra_blocks is not None:
            results, names = self.extra_blocks(results, x, names)

        # make it back an OrderedDict
        out = OrderedDict([(k, self.dequants[i](v)) for i, (k, v) in enumerate(zip(names, results))])

        return out


class BackboneWithFPN(backbone_utils.BackboneWithFPN):
    """
    Reimplementation of `torchvision.models.detection.backbone_utils.BackboneWithFPN` to use modified FeaturePyramidNetwork.
    """
    def __init__(
        self,
        backbone: torch.nn.Module,
        return_layers: tp.Dict[str, str],
        in_channels_list: tp.List[int],
        out_channels: int,
        extra_blocks: tp.Optional[feature_pyramid_network.ExtraFPNBlock] = None,
        norm_layer: tp.Optional[tp.Callable[..., torch.nn.Module]] = None,
    ) -> None:
        
        # Initialize Original BackboneWithFPN
        super().__init__(backbone, return_layers, in_channels_list, out_channels, extra_blocks, norm_layer)

        # Overwrite the FeaturePyramidNetwork
        self.fpn = FeaturePyramidNetwork(
            in_channels_list=in_channels_list,
            out_channels=out_channels,
            extra_blocks=extra_blocks,
            norm_layer=norm_layer,
            return_layers=return_layers
        )


def _resnet_fpn_extractor(
    backbone: resnet.ResNet,
    trainable_layers: int,
    returned_layers: tp.Optional[tp.List[int]] = None,
    extra_blocks: tp.Optional[feature_pyramid_network.ExtraFPNBlock] = None,
    norm_layer: tp.Optional[tp.Callable[..., torch.nn.Module]] = None,
) -> BackboneWithFPN:
    
    """
    Reimplementation of `torchvision.models.detection.backbone_utils._resnet_fpn_extractor` to use modified BackboneWithFPN.

    Raises:
        ValueError: _description_
        ValueError: _description_

    Returns:
        BackboneWithFPN: _description_
    """

    # select layers that wont be frozen
    if trainable_layers < 0 or trainable_layers > 5:
        raise ValueError(f"Trainable layers should be in the range [0,5], got {trainable_layers}")
    layers_to_train = ["layer4", "layer3", "layer2", "layer1", "conv1"][:trainable_layers]
    if trainable_layers == 5:
        layers_to_train.append("bn1")
    for name, parameter in backbone.named_parameters():
        if all([not name.startswith(layer) for layer in layers_to_train]):
            parameter.requires_grad_(False)

    if extra_blocks is None:
        extra_blocks = feature_pyramid_network.LastLevelMaxPool()

    if returned_layers is None:
        returned_layers = [1, 2, 3, 4]
    if min(returned_layers) <= 0 or max(returned_layers) >= 5:
        raise ValueError(f"Each returned layer should be in the range [1,4]. Got {returned_layers}")
    return_layers = {f"layer{k}": str(v) for v, k in enumerate(returned_layers)}

    in_channels_stage2 = backbone.inplanes // 8
    in_channels_list = [in_channels_stage2 * 2 ** (i - 1) for i in returned_layers]
    out_channels = 256
    return BackboneWithFPN(
        backbone, return_layers, in_channels_list, out_channels, extra_blocks=extra_blocks, norm_layer=norm_layer
    )


def fuse_faster_rcnn(model: torch.nn.Module):
    torch.ao.quantization.fuse_modules(model.rpn.head.conv[0], ['0', '1'], inplace=True)


def fuse_resnet(model: torch.nn.Module, backbone: str, quantizable: bool = False):
    """
    Fuse resnet backbone modules of a model with backbone.

    Args:
        model (torch.nn.Module): Model with resnet backbone
        backbone (str): Name of resnet backbone
        quantizable (bool, optional): If backbone is quantizable or not. Defaults to False.

    Raises:
        NotImplementedError: _description_
        NotImplementedError: _description_
    """
    torch.ao.quantization.fuse_modules(model.backbone.body, ['conv1', 'bn1', 'relu'], inplace=True)


    basic_block = 'QuantizableBasicBlock' if quantizable else 'BasicBlock'
    bottleneck = 'QuantizableBottleneck' if quantizable else 'Bottleneck'

    if quantizable:
        bottleneck_modules = [['conv1', 'bn1', 'relu1'], ['conv2', 'bn2', 'relu2'], ['conv3', 'bn3']]
        basic_block_modules =  [['conv1', 'bn1', 'relu'], ['conv2', 'bn2']]
    else:
        bottleneck_modules = [['conv1', 'bn1'], ['conv2', 'bn2'], ['conv3', 'bn3']]
        basic_block_modules =  [['conv1', 'bn1',], ['conv2', 'bn2']]

    
    if backbone == 'resnet50':
        layers = ['layer1', 'layer2', 'layer3', 'layer4']

        for layer in layers:
            for i in range(len(model.backbone.body[layer])):
                if type(model.backbone.body[layer][i]).__name__ ==  bottleneck:
                    modules = bottleneck_modules

                elif type(model.backbone.body[layer][i]).__name__ ==  basic_block:
                    modules = basic_block_modules

                else:
                    raise NotImplementedError
                
                torch.ao.quantization.fuse_modules(model.backbone.body[layer][i], modules, inplace=True)

                if model.backbone.body[layer][i].downsample is not None:
                    if isinstance(model.backbone.body[layer][i].downsample[0], torch.nn.Conv2d)   and isinstance(model.backbone.body[layer][i].downsample[1], torch.nn.BatchNorm2d):
                        torch.ao.quantization.fuse_modules(model.backbone.body[layer][i].downsample, ['0', '1'], inplace=True)
     
    else:
        raise NotImplementedError


def fuse_modules(model: torch.nn.Module, backbone: str = 'resnet50', quantizable: bool = False):
    """
    Fuse model modules with backbone

    Args:
        model (torch.nn.Module): Model with backbone
        backbone (str, optional): Backbone name. Defaults to 'resnet50'.
        quantizable (bool, optional): if backbone is quantizable or not. Defaults to False.

    Raises:
        NotImplementedError: _description_
    """
    if 'resnet' in backbone:
        fuse_resnet(model, backbone=backbone, quantizable=quantizable)
    else:
        raise NotImplementedError


def prepare_faster_rcnn(model: torch.nn.Module, backbone: str, quantizable_backbone: bool) -> torch.nn.Module:
    """
    Prepare faster rcnn for quantization.
    1. If the backbone was not quantizable replace with a quantizable version
    2. Fuse modules
    3. return model

    Args:
        model (torch.nn.Module): faster rcnn model
        backbone (str): backbone name
        quantizable_backbone (bool): if backbone is quantizable or not

    Returns:
        torch.nn.Module: prepared model
    """

    if not quantizable_backbone:
        if backbone == 'resnet50':
            quantizable_backbone = resnet.resnet50(weights=None, progress=False, norm_layer=torch.nn.BatchNorm2d)
        elif backbone == 'resnet34':
            quantizable_backbone = resnet.resnet34(weights=None, progress=False, norm_layer=torch.nn.BatchNorm2d)
        elif backbone == 'resnet18':
            quantizable_backbone = resnet.resnet18(weights=None, progress=False, norm_layer=torch.nn.BatchNorm2d)

        
        quantizable_backbone = _resnet_fpn_extractor(quantizable_backbone, 5).eval()

        # Now load already present backbones weights here
        quantizable_backbone.load_state_dict(model.backbone.state_dict())
        model.backbone = quantizable_backbone

    
    # Eager module does not fuse eligible modules automatically.
    fuse_modules(model, backbone=backbone, quantizable=True)
    
    return model


def prepare_model_eager(model: torch.nn.Module,
                        quantization_scheme: tp.Optional[torch.qscheme] = None,
                        backend: str = 'fbgemm', insert_dequan_stub: bool = False) -> torch.nn.Module:
    
    """
    Prepare model for quantization in eager mode. Insert quant/dequant stubs and call torch.quantization.prepare
    NOTE: model is replaced in place

    Raises:
        NotImplementedError: _description_

    Returns:
        torch.nn.Module: model ready for quantization.
    """

    # Insert quantization stubs
    if insert_dequan_stub:
        model = torch.nn.Sequential(torch.quantization.QuantStub(), 
                    model, 
                    torch.quantization.DeQuantStub())
    else:
        model = torch.nn.Sequential(torch.quantization.QuantStub(), 
                    model)

    if quantization_scheme is None:
        model.qconfig = torch.quantization.get_default_qconfig(backend=backend)
    else:
        raise NotImplementedError

    torch.quantization.prepare(model, inplace=True)

    return model
    

def caliberate(backbone: torch.nn.Module, dataloader: Dataloader, device: str ='cuda', transform: tp.Optional[tp.Callable] = None):
    import tqdm

    backbone = backbone.to(device)
    with torch.inference_mode():
        for batch in tqdm.tqdm(dataloader):
            if transform is not None:
                batch = transform(batch)
            batch = batch.to(device)
            backbone(batch)


def quantize_model(model: torch.nn.Module,
                   backbone:str,
                   dataloader: Dataloader,
                   mode: tp.Literal['eager', 'fx_graph'],
                   quantizable_backbone: bool = False,
                   inplace: bool = False,
                   inser_dequant_stub: bool = False,
                   quantize_only_backbone: bool = True,
                   transform: tp.Optional[tp.Callable] = None) -> torch.nn.Module:

    model = prepare_faster_rcnn(model, backbone=backbone, quantizable_backbone=quantizable_backbone)


    if mode == 'eager':
        if quantize_only_backbone:
            backbone = prepare_model_eager(model.backbone, inser_dequant_stub=inser_dequant_stub)
            caliberate(backbone, dataloader=dataloader, transform=transform)
            backbone = torch.quantization.convert(backbone, inplace=inplace)
            model.backbone = backbone
        else:
            raise NotImplementedError
    
    else:
        raise NotImplementedError
    
    return model


def resnet_fpn_bn(
    *,
    progress: bool = False,
    num_classes: tp.Optional[int] = 2,
    weights_backbone: tp.Union[ResNet18_Weights, ResNet34_Weights, ResNet50_Weights] = ResNet50_Weights.IMAGENET1K_V1,
    trainable_backbone_layers: int = 3,
    **kwargs: tp.Any,
) -> faster_rcnn.FasterRCNN:
    '''Copied from https://github.com/pytorch/vision/blob/2c44ebaeece31b0cc9a7385e406312f741333ab5/torchvision/models/detection/faster_rcnn.py#L459
       Modifications:
        - BatchNorm instead of FrozenBatchNorm
    '''
    #MODIFIED: always BatchNorm
    norm_layer = torch.nn.BatchNorm2d

    is_trained = True
    trainable_backbone_layers = backbone_utils._validate_trainable_layers(is_trained, trainable_backbone_layers, 6, 3)

    if isinstance(weights_backbone, ResNet18_Weights):
        backbone = torchvision.models.resnet18(weights=weights_backbone, progress=progress, norm_layer=norm_layer)
    elif isinstance(weights_backbone, ResNet34_Weights):
        backbone = torchvision.models.resnet34(weights=weights_backbone, progress=progress, norm_layer=norm_layer)
    elif isinstance(weights_backbone, ResNet50_Weights):
        backbone = torchvision.models.resnet50(weights=weights_backbone, progress=progress, norm_layer=norm_layer)


    backbone = backbone_utils._resnet_fpn_extractor(backbone, trainable_backbone_layers)

    # These are the features returned when `returned_layers` is not set.
    # Resnet backbones return features from all layers except `conv0` so 4
    featmap_names = ["0", "1", "2", "3"]
    anchor_sizes = ((32,), (64,), (128,), (256,), (512,))
    aspect_ratios = ((0.5, 1.0, 2.0),) * len(anchor_sizes)

    box_roi_pool = torchvision.ops.MultiScaleRoIAlign(
                featmap_names=featmap_names, output_size=7, sampling_ratio=2
            )
    model = faster_rcnn.FasterRCNN(
        backbone, num_classes, rpn_anchor_generator=anchor_utils.AnchorGenerator(anchor_sizes, aspect_ratios), box_roi_pool=box_roi_pool, **kwargs
    )

    return model
