import torch
import torchvision
from torchvision import transforms

_dataset_name = ["default", "cifar10", "gtsrb", "imagenet"]

_mean = {
    "default":  [0.5, 0.5, 0.5],
    "cifar10":  [0.4914, 0.4822, 0.4465],
    "gtsrb":    [0.3337, 0.3064, 0.3171],
    "imagenet": [0.485, 0.456, 0.406],
}

_std = {
    "default":  [0.5, 0.5, 0.5],
    "cifar10":  [0.2470, 0.2435, 0.2616],
    "gtsrb":    [0.2672, 0.2564, 0.2629],
    "imagenet": [0.229, 0.224, 0.225],
}

_size = {
    "cifar10":  (32, 32),
    "gtsrb":    (32, 32),
    "imagenet": (224, 224),
}


def get_totensor_topil():
    return transforms.ToTensor(), transforms.ToPILImage()

def get_normalize_unnormalize(dataset):
    assert dataset in _dataset_name, _dataset_name
    mean = torch.FloatTensor(_mean[dataset])
    std = torch.FloatTensor(_std[dataset])
    normalize = transforms.Normalize(mean, std)
    unnormalize = transforms.Normalize(- mean / std, 1 / std)
    return normalize, unnormalize

def get_clip_normalized(dataset):
    normalize, _ = get_normalize_unnormalize(dataset)
    return lambda x : torch.min(torch.max(x, normalize(torch.zeros_like(x))), normalize(torch.ones_like(x)))

def get_resize(size):
    if isinstance(size, str):
        assert size in _dataset_name, "'size' should be (width, height) or dataset name. Available dataset name:" + str(_dataset_name)
        size = _size[size]
    return transforms.Resize(size)

def get_preprocess_deprocess(dataset, size=None):
    """
    :param size: (width, height) or dataset name
    """
    totensor, topil = get_totensor_topil()
    normalize, unnormalize = get_normalize_unnormalize(dataset)
    if size is None:
        preprocess = transforms.Compose([totensor, normalize])
        deprocess = transforms.Compose([unnormalize, topil])
    else:
        preprocess = transforms.Compose([get_resize(size), totensor, normalize])
        deprocess = transforms.Compose([unnormalize, topil])
    return preprocess, deprocess
    