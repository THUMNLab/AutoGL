import numpy as np
import pickle

class EarlyStopping(object):
    def __init__(self, patience=10):
        self.patience = patience
        self.counter = 0
        self.best_acc = None
        self.best_loss = None
        self.early_stop = False
        self.model = None

    def step(self, loss, acc, model):
        if self.best_loss is None:
            self.best_acc = acc
            self.best_loss = loss
            self.save_checkpoint(model)
        elif (loss > self.best_loss) and (acc < self.best_acc):
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            if (loss <= self.best_loss) and (acc >= self.best_acc):
                self.save_checkpoint(model)
            self.best_loss = np.min((loss, self.best_loss))
            self.best_acc = np.max((acc, self.best_acc))
            self.counter = 0
        return self.early_stop

    def save_checkpoint(self, model):
        """Saves model when validation loss decreases."""
        self.model = pickle.dumps(model.state_dict())

    def load_checkpoint(self, model):
        """Load the latest checkpoint."""
        model.load_state_dict(pickle.loads(self.model))

def get_encoder_decoder_hp(model='han'):
    if model == "han":
        return {
            "num_layers": 2,
            "hidden": [256], ##
            "heads": [8], ##
            "dropout": 0.2,
            "act": "gelu",
        }, None
    if model == "hgt":
        return {
            "num_layers": 2,
            "hidden": [256,256,256],
            "heads": 4,
            "dropout": 0.2,
            "act": "gelu",
            "use_norm": True,
        }, None
    if model == "HeteroRGCN":
        return {
            "num_layers": 2,
            "hidden": [256],
            "heads": 4,
            "dropout": 0.2,
            "act": "leaky_relu",
        }, None
    return {}, None