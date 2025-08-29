import yaml
from trainers.train import Trainer

if __name__ == "__main__":
    config = yaml.safe_load(open("configs/oasis.yaml"))
    trainer = Trainer(config)
    trainer.run()

