import sys, os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import yaml

from trainers.train import Trainer
from datasets.oasis_dataset import OASISDataset
from utils.preprocess import preprocess_data
 
from trainers.train import Trainer

if __name__ == "__main__":
    config = yaml.safe_load(open("configs/oasis.yaml"))
    trainer = Trainer(config)
    trainer.run()

