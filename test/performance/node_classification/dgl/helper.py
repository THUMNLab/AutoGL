def get_encoder_decoder_hp(model='gin', decoder=None):
    if model == 'gin':
        model_hp = {
            "num_layers": 5,
            "hidden": [64],
            "act": "relu",
            "eps": "False",
            "mlp_layers": 2,
            "neighbor_pooling_type": "sum"
        }
    elif model == 'gat':
        model_hp = {
            # hp from model
            "num_layers": 2,
            "hidden": [8],
            "heads": 8,
            "dropout": 0.6,
            "act": "relu",
        }
    elif model == 'gcn':
        model_hp = {
            "num_layers": 2,
            "hidden": [16],
            "dropout": 0.5,
            "act": "relu"
        }
    elif model == 'sage':
        model_hp = {
            "num_layers": 2,
            "hidden": [64],
            "dropout": 0.5,
            "act": "relu",
            "agg": "gcn",
        }
    elif model == 'topk':
        model_hp = {
            "num_layers": 5,
            "hidden": [64, 64, 64, 64]
        }
    
    return model_hp, {}
