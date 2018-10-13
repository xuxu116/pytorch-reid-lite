import resnet


network_fn = {
    "resnet_18": resnet.resnet18,
    "resnet_34": resnet.resnet34,
    "resnet_50": resnet.resnet50,
    "resnet_101": resnet.resnet101,
    "resnet_152": resnet.resnet152
}
