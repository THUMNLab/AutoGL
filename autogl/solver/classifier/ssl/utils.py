from ....module.model import EncoderUniversalRegistry, DecoderUniversalRegistry, ModelUniversalRegistry

def _parse_hp_space(spaces):
    if spaces is None:
        return None
    for space in spaces:
        if "cutFunc" in space and isinstance(space["cutFunc"], str):
            space["cutFunc"] = eval(space["cutFunc"])
    return spaces

def _parse_model_hp(model):
    assert isinstance(model, dict)
    if "encoder" in model and "decoder" in model:
        if "prediction_head" in model:
            return {
                "encoder": _parse_hp_space(model["encoder"].pop("hp_space", None)),
                "decoder": _parse_hp_space(model["decoder"].pop("hp_space", None)),
                "prediction_head": _parse_hp_space(model["prediction_head"].pop("hp_space", None))
            }
        else:
            return {
                "encoder": _parse_hp_space(model["encoder"].pop("hp_space", None)),
                "decoder": _parse_hp_space(model["decoder"].pop("hp_space", None)),
            }
    elif "encoder" in model:
        return {
            "encoder": _parse_hp_space(model["encoder"].pop("hp_space", None)),
            "decoder": None,
        }
    else:
        return _parse_hp_space(model.pop("hp_space", None))

def _initialize_single_model(model):
    encoder, decoder = None, None
    if "encoder" in model:
        # initialize encoder
        name = model["encoder"].pop("name")
        encoder = EncoderUniversalRegistry.get_encoder(name)(**model["encoder"])
    if "decoder" in model:
        # initialize decoder
        name = model["decoder"].pop("name")
        decoder = DecoderUniversalRegistry.get_decoder(name)(**model["decoder"])
        return (encoder, decoder)

    if "name" in model:
        # whole model
        name = model.pop("name")
        encoder = ModelUniversalRegistry.get_model(name)(**model)
    return encoder
