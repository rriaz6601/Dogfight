import torch


def calc_num_elements(module, module_input_shape):
    """

    Parameters
    ----------
    module :

    module_input_shape :


    Returns
    -------

    """
    shape_with_batch_dim = (1,) + module_input_shape
    some_input = torch.rand(shape_with_batch_dim)
    num_elements = module(some_input).numel()
    return num_elements


def to_scalar(value):
    """

    Parameters
    ----------
    value :


    Returns
    -------

    """
    if isinstance(value, torch.Tensor):
        return value.item()
    else:
        return value
