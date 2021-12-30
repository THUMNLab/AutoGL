def get_encoder_decoder_hp(model='gat', decoder='lpdecoder'):
    if model == 'gat':
        model_hp = {
            "num_layers": 2,
            "hidden": [16, 16],
            "dropout": 0.0,
            "act": "relu",
            "num_hidden_heads": 8,
            "num_output_heads": 8
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
            "agg": "mean"
        }

    return model_hp, {}
