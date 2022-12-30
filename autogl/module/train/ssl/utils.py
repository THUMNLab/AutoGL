from .views_fn import (
    DropNode,
    PermuteEdge,
    MaskNode,
    SubGraph,
    RandomView
)

def get_view_by_name(view, aug_ratio):
    if view is None:
        return lambda x: x
    elif view == "dropN":
        return DropNode(aug_ratio=aug_ratio)
    elif view == "permE":
        return PermuteEdge(aug_ratio=aug_ratio)
    elif view == "subgraph":
        return SubGraph(aug_ratio=aug_ratio)
    elif view == "maskN":
        return MaskNode(aug_ratio=aug_ratio)
    elif view == "random2":
        canditates = [DropNode(aug_ratio=aug_ratio),
                      SubGraph(aug_ratio=aug_ratio)]
        return RandomView(candidates=canditates)
    elif view == "random3":
        canditates = [DropNode(aug_ratio=aug_ratio),
                      SubGraph(aug_ratio=aug_ratio),
                      PermuteEdge(aug_ratio=aug_ratio)]
        return RandomView(candidates=canditates)
    elif view == "random4":
        canditates = [DropNode(aug_ratio=aug_ratio),
                      SubGraph(aug_ratio=aug_ratio),
                      PermuteEdge(aug_ratio=aug_ratio),
                      MaskNode(aug_ratio=aug_ratio)]
        return RandomView(candidates=canditates)
    else:
        raise NotImplementedError(f'{view} is not supported yet. Support: ["dropN", "permE", "subgraph", \
                                    "maskN", "random2", "random3", "random4", None]')
