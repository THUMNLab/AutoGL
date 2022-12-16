from dig.sslgraph.method.contrastive.views_fn import (
    NodeAttrMask,
    EdgePerturbation,
    UniformSample,
    RWSample,
    RandomView
)

def get_view_by_name(view, aug_ratio):
    if view is None:
        return lambda x: x
    elif view == "dropN":
        return UniformSample(ratio=aug_ratio)
    elif view == "permE":
        return EdgePerturbation(ratio=aug_ratio)
    elif view == "subgraph":
        return RWSample(ratio=aug_ratio)
    elif view == "maskN":
        return NodeAttrMask(mask_ratio=aug_ratio)
    elif view == "random2":
        canditates = [UniformSample(ratio=aug_ratio),
                      RWSample(ratio=aug_ratio)]
        return RandomView(candidates=canditates)
    elif view == "random3":
        canditates = [UniformSample(ratio=aug_ratio),
                      RWSample(ratio=aug_ratio),
                      EdgePerturbation(ratio=aug_ratio)]
        return RandomView(candidates=canditates)
    elif view == "random4":
        canditates = [UniformSample(ratio=aug_ratio),
                      RWSample(ratio=aug_ratio),
                      EdgePerturbation(ratio=aug_ratio),
                      NodeAttrMask(mask_ratio=aug_ratio)]
        return RandomView(candidates=canditates)
    else:
        raise NotImplementedError(f'The augmentation method must be in ["dropN", "permE", "subgraph", \
                                    "maskN", "random2", "random3", "random4"] or None. And {view} is not supported yet.')