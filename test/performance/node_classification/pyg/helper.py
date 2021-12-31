def get_encoder_decoder_hp(model='gin', decoder=None, decoupled=False):
    if model == 'gin':
        model_hp = {
            # hp from model
            "num_layers": 2,
            "hidden": [64],
            "dropout": 0.5,
            "act": "relu",
            "eps": "False",
            "mlp_layers": 2
        }

    if model == 'gat':
        if decoupled:
            model_hp = {
                "num_layers": 2,
                "hidden": [8],
                "num_hidden_heads": 8,
                "num_output_heads": 1,
                "dropout": 0.6,
                "act": "elu"
            }
        else:
            model_hp = {
                "num_layers": 2,
                "hidden": [8],
                "heads": 8,
                "dropout": 0.6,
                "act": "elu"
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
            "agg": "mean",
        }
    else:
        model_hp = {}
    
    return model_hp, {}
