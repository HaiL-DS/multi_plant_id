import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader, random_split
import os


os.chdir("/scratch/ezq9qu/images_max_side_800")
DATA_DIR = os.getcwd()


RESIZE_SIZE = 256
IMG_SIZE = 224 
BATCH_SIZE = 32
NUM_CLASSES = 7806 # As per the challenge overview
NUM_WORKERS = os.cpu_count() # Use all available CPU cores for loading


class SinglePlantDataLoader:
    """
    A class to create training, validation, and test DataLoaders
    from a single directory containing subfolders for each class.
    Splits the data based on 80/10/10 Train/Test/Val
    """
    def __init__(self, data_dir, resize_size, img_size, batch_size, num_workers, train_split=0.8, val_split=0.1, test_split=0.1):
        """
        Initializes the SinglePlantDataLoader.

        Args:
            data_dir (str): Path to the root directory containing class subfolders.
            resize_size (int): The size to which the smaller edge of the image will be resized.
            img_size (int): The size of the center crop or random resized crop.
            batch_size (int): The number of images per batch.
            num_workers (int): How many subprocesses to use for data loading.
            train_split (float): The proportion of the dataset to use for training.
            val_split (float): The proportion of the dataset to use for validation.
            test_split (float): The proportion of the dataset to use for testing.
        """
        self.data_dir = data_dir
        self.resize_size = resize_size
        self.img_size = img_size
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_split = train_split
        self.val_split = val_split
        self.test_split = test_split

        self.train_loader = None
        self.val_loader = None
        self.test_loader = None
        self.classes = None

        self._load_and_split_data()


    def _load_and_split_data(self):
        """
        Loads the dataset, splits it, and creates the DataLoaders.
        """
        # Define the standard PyTorch transforms for ImageNet-pretrained models
        val_test_transform = transforms.Compose([
            transforms.Resize(self.resize_size),
            transforms.CenterCrop(self.img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(self.img_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

        print(f"Loading data from: {self.data_dir}")
        try:
            # Create the full dataset
            full_dataset = datasets.ImageFolder(
                root=self.data_dir,
                transform=val_test_transform # Use validation transform for initial loading
            )

            # Check if dataset is empty
            if not full_dataset.samples:
                print(f"ERROR: No images found in {self.data_dir}.")
                return

            self.classes = full_dataset.classes


        except Exception as e:
            print(f"An unexpected error occurred loading data: {e}")
            return

        # --- Split dataset into train, val, and test ---
        dataset_size = len(full_dataset)
        train_size = int(self.train_split * dataset_size)
        val_size = int(self.val_split * dataset_size)
        test_size = dataset_size - train_size - val_size # Ensure all data is used

        print(f"\nSplitting dataset")

        train_dataset, val_dataset, test_dataset = random_split(
            full_dataset,
            [train_size, val_size, test_size]
        )

        # Apply the training transform to the training split
        train_dataset.dataset.transform = train_transform




        print(f"\nFound {len(full_dataset)} total images belonging to {len(self.classes)} classes.")

        # Create the DataLoaders
        self.train_loader = DataLoader(
            dataset=train_dataset,
            batch_size=self.batch_size,
            shuffle=True,      # Shuffle the training data
            num_workers=self.num_workers,
            pin_memory=True    # Speeds up CPU-to-GPU data transfer
        )

        self.val_loader = DataLoader(
            dataset=val_dataset,
            batch_size=self.batch_size,
            shuffle=False,     
            num_workers=self.num_workers,
            pin_memory=True
        )

        self.test_loader = DataLoader(
            dataset=test_dataset,
            batch_size=self.batch_size,
            shuffle=False,     
            num_workers=self.num_workers,
            pin_memory=True
        )

    def get_dataloaders(self):
        """
        Returns the training, validation, and test DataLoaders.

        Returns:
            tuple: (train_loader, val_loader, test_loader)
        """
        return self.train_loader, self.val_loader, self.test_loader

    def get_classes(self):
        """
        Returns the list of class names found in the dataset.

        Returns:
            list: A list of class names.
        """
        return self.classes


def main():
    """
    Main function
    """

    # Create an instance of the SinglePlantDataLoader
    data_splitter = SinglePlantDataLoader(
        data_dir=DATA_DIR,
        resize_size=RESIZE_SIZE,
        img_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS
    )

    # Get the dataloaders
    train_loader, val_loader, test_loader = data_splitter.get_dataloaders()

    if train_loader is None or val_loader is None or test_loader is None:
        return

    print("\n--- Dataloader Pipelines Created ---")


if __name__ == "__main__":
    main()