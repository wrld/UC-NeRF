import imp
from .scared import ScaredDataset
from .hamlyn import HamlynDataset
dataset_dict = {
                'scared': ScaredDataset,
                'hamlyn': HamlynDataset}