import torch
import torchvision
import numpy.testing as np_test

from src import quantization


def test_fuse_modules():
    detection_model = quantization.resnet_fpn_bn(weights=None, num_classes=2)

    detection_model = detection_model.eval()
    x = torch.rand((1, 3, 512, 512), dtype=torch.float32)

    with torch.inference_mode():
        original_outs = detection_model(x)[0]
        original_outs = (util.to_numpy(original_outs['boxes']), util.to_numpy(original_outs['labels']), util.to_numpy(original_outs['scores']))

    
    quantization.fuse_modules(detection_model, backbone='resnet50', quantizable=False)
    with torch.inference_mode():
        fused_outs = detection_model(x)[0]
        fused_outs = (util.to_numpy(fused_outs['boxes']), util.to_numpy(fused_outs['labels']), util.to_numpy(fused_outs['scores']))

    # boxes test
    np_test.assert_allclose(original_outs[0], fused_outs[0], atol=1e-05, rtol=1e-03)
    # labels test
    np_test.assert_allclose(original_outs[1], fused_outs[1], atol=1e-05, rtol=1e-03)
    # scores test
    np_test.assert_allclose(original_outs[2], fused_outs[2], atol=1e-05, rtol=1e-03)


def test_quantizable_modules_swap():
    detection_model = 
    detection_model = detection_model.eval()
    x = torch.rand((1, 3, 512, 512), dtype=torch.float32)

    with torch.inference_mode():
        original_outs = detection_model(x)[0]
        original_outs = (util.to_numpy(original_outs['boxes']), util.to_numpy(original_outs['labels']), util.to_numpy(original_outs['scores']))

    # Create a backbone with fpn with quantizable ops
    backbone = torchvision.models.resnet50(weights=None, progress=False, norm_layer=torch.nn.BatchNorm2d)
    backbone_fpn = quantization._resnet_fpn_extractor(backbone, trainable_layers=5).eval()

    backbone_fpn.load_state_dict(detection_model.backbone.state_dict())

    # swap backbones
    detection_model.backbone = backbone_fpn

    with torch.inference_mode():
        test_outs = detection_model(x)[0]
        test_outs = (util.to_numpy(test_outs['boxes']), util.to_numpy(test_outs['labels']), util.to_numpy(test_outs['scores']))

    # boxes test
    np_test.assert_allclose(original_outs[0], test_outs[0], atol=1e-05, rtol=1e-03)
    # labels test
    np_test.assert_allclose(original_outs[1], test_outs[1], atol=1e-05, rtol=1e-03)
    # scores test
    np_test.assert_allclose(original_outs[2], test_outs[2], atol=1e-05, rtol=1e-03)
