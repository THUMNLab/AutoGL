def get_encoder_decoder_hp(model='gin', decoder=None, decoupled=True):
    if model == 'gat':
        if decoupled:
            model_hp = {
                "num_layers": 3,
                "hidden": [8, 8],
                "num_hidden_heads": 8,
                "num_output_heads": 8,
                "dropout": 0.0,
                "act": "relu"
            }
        else:
            model_hp = {
                "num_layers": 3,
                "hidden": [8, 8],
                "heads": 8,
                "dropout": 0.0,
                "act": "relu"
            }
    elif model == 'gcn':
        model_hp = {
            "num_layers": 3,
            "hidden": [16, 16],
            "dropout": 0.,
            "act": "relu",
        }
    elif model == 'sage':
        model_hp = {
            'num_layers': 3,
            'hidden': [16, 16],
            'dropout': 0.0,
            'act': 'relu',
            'agg': 'mean'
        }
        
    return model_hp, {}
