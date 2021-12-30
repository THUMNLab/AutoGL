def get_encoder_decoder_hp(model='gat', decoder='lpdecoder'):
    if model == 'gat':
        model_hp = {
            # hp from model
            "num_layers": 3,
            "hidden": [16,64],
            "heads": 8,
            "dropout": 0.0,
            "act": "relu",
            'add_self_loops': 'False',
            'normalize': 'False',
        }
    elif model == 'gcn':
        model_hp = {
            "hidden": [128, 64],
            "dropout": 0.0,
            "act": "relu"
        }
    elif model == 'sage':
        model_hp = {
            "num_layers": 3,
            "hidden": [128,64],
            "dropout": 0.0,
            "act": "relu",
            "agg": "mean",
            'add_self_loops': 'False',
            'normalize': 'False',
        }
        
    return model_hp, {}
