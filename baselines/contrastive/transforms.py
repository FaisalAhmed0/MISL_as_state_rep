from torchvision import transforms


class Transform:
    def __init__(self):
        self.transform = transforms.Compose([
          transforms.RandomResizedCrop(84),
          transforms.RandomHorizontalFlip(p=0.5),
          # transforms.RandomApply(
          #     [transforms.ColorJitter(brightness=0.4, contrast=0.4,
          #                             saturation=0.2, hue=0.1)],
          #     p=0.8
          # ),
          # transforms.RandomGrayscale(p=0.2),
#           transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5)),
          # Solarization(p=0.0),
#           transforms.ToTensor(),
#           transforms.Normalize(
#                   mean=(0.5)*9, std=(0.5)*9)
        ]
        )
        self.transform_prime = transforms.Compose([
          transforms.RandomResizedCrop(84),
          transforms.RandomHorizontalFlip(p=0.5),
          # transforms.RandomApply(
          #     [transforms.ColorJitter(brightness=0.4, contrast=0.4,
          #                             saturation=0.2, hue=0.1)],
          #     p=0.8
          # ),
          # transforms.RandomGrayscale(p=0.2),
#           transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5)),
          # Solarization(p=0.2),
#           transforms.ToTensor(),
#           transforms.Normalize(
#                   mean=(0.5)*9, std=(0.5)*9)
        ]
        )

    def __call__(self, x):
        y1 = self.transform(x)
        y2 = self.transform_prime(x)
        return y1, y2