def get_encoder_decoder_hp(model='gin', decoder=None):
    if model == 'gin':
        model_hp = {
            # hp from model
            "num_layers": 5,
            "hidden": [64,64,64,64],
            "dropout": 0.5,
            "act": "relu",
            "eps": "False",
            "mlp_layers": 2
        }

    if model == 'gat':
        model_hp = {
            "num_layers": 2,
            "hidden": [8],
            "num_hidden_heads": 8,
            "num_output_heads": 8,
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
    
    if decoder is None or decoder == "addpoolmlp":
        decoder_hp = {
            "hidden": 64,
            "act": "relu",
            "dropout": 0.5
        }
    elif decoder == "diffpool":
        decoder_hp = {
            "ratio": 0.8,
            "dropout": 0.5,
            "act": "relu"
        }
    else:
        decoder_hp = {}
    
    return model_hp, decoder_hp
