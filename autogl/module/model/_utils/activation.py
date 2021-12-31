import torch.nn.functional
import typing as _typing


def activation_func(
        tensor: torch.Tensor, function_name: _typing.Optional[str]
) -> torch.Tensor:
    if not isinstance(function_name, str):
        return tensor
    elif function_name == 'linear':
        return tensor
    elif function_name == 'tanh':
        return torch.tanh(tensor)
    elif hasattr(torch.nn.functional, function_name):
        return getattr(torch.nn.functional, function_name)(tensor)
    else:
        raise TypeError(f"PyTorch does not support activation function {function_name}")
