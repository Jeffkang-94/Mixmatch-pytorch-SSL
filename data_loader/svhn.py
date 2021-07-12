import torchvision

class Labeled(torchvision.datasets):
    def __init__(self, root, indexs=None, train=True,
                    transform=None, target_transform=None, download=False):
        super(Labeled, self).__init__(root=root, train=train, transform=transform,
                                    target_transform=target_transform, download=download)
        dataset = 