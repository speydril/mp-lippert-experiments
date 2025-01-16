import src.util.transforms_shir as transforms


def get_polyp_transform(mean, std):
    transform_train = transforms.Compose(
        [
            # transforms.Resize((352, 352)),
            transforms.ToPILImage(),
            # transforms.ColorJitter(
            #     brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1  # type: ignore
            # ),
            transforms.RandomVerticalFlip(),
            transforms.RandomHorizontalFlip(),
            transforms.RandomAffine(90, scale=(0.75, 1.25)),
            transforms.CLAHE(),
            transforms.RandomContrast(),
            transforms.RandomGamma(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )
    transform_test = transforms.Compose(
        [
            # transforms.Resize((352, 352)),
            transforms.ToPILImage(),
            transforms.CLAHE(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )
    return transform_train, transform_test
