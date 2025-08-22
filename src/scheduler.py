from torch.optim.lr_scheduler import LambdaLR
from functools import partial
from torch.optim import Optimizer
from typing import Optional, Dict, Any
from transformers.optimization import get_cosine_schedule_with_warmup

def get_scheduler(
    name: str,
    optimizer: Optimizer,
    num_warmup_steps: Optional[int] = None,
    num_training_steps: Optional[int] = None,
    scheduler_specific_kwargs: Optional[dict] = None,
    ):
    if name == "constant":
        return get_constant_schedule_with_warmup(optimizer, num_warmup_steps, **(scheduler_specific_kwargs or {}))
    elif name == "wsd":
        return get_stable_decay_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, **(scheduler_specific_kwargs or {}))
    elif name == "cosine":
        return get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, **(scheduler_specific_kwargs or {}))
    else:
        raise ValueError(f"Unknown scheduler name: {name}")


def get_constant_schedule_with_warmup(optimizer: Optimizer, num_warmup_steps: int, last_epoch: int = -1):
    """
    Create a schedule with a constant learning rate preceded by a warmup period during which the learning rate
    increases linearly between 0 and the initial lr set in the optimizer.

    Args:
        optimizer ([`~torch.optim.Optimizer`]):
            The optimizer for which to schedule the learning rate.
        num_warmup_steps (`int`):
            The number of steps for the warmup phase.
        last_epoch (`int`, *optional*, defaults to -1):
            The index of the last epoch when resuming training.

    Return:
        `torch.optim.lr_scheduler.LambdaLR` with the appropriate schedule.
    """

    lr_lambda = partial(_get_constant_schedule_with_warmup_lr_lambda, num_warmup_steps=num_warmup_steps)
    return LambdaLR(optimizer, lr_lambda, last_epoch=last_epoch)

def _get_constant_schedule_with_warmup_lr_lambda(current_step: int, num_warmup_steps: int):
    if current_step < num_warmup_steps:
        return float(current_step) / float(max(1, num_warmup_steps))
    return 1.0

def get_stable_decay_schedule_with_warmup(optimizer: Optimizer, num_warmup_steps: int, num_training_steps:int,  num_decay_steps: int = None, last_epoch: int = -1):
    """
    Create a schedule that keeps the learning rate constant, setting a warm-up period before it and a decay period after it.
    During the warm-up period, the learning rate increases linearly from 0 to the initial learning rate (lr) set by the optimizer.
    During the decay period, the learning rate decreases linearly from the initial learning rate (lr) set by the optimizer to 0.

    Args:
        optimizer ([`~torch.optim.Optimizer`]):
            The optimizer for which to schedule the learning rate.
        num_warmup_steps (`int`):
            The number of steps for the warmup phase.
        num_training_steps (`int`):
            The total number of training steps.
        num_decay_steps (`int`, *optional*):
            The number of steps for the decay phase. defaults to 2*num_warmup_steps
        last_epoch (`int`, *optional*, defaults to -1):
            The index of the last epoch when resuming training.

    Return:
        `torch.optim.lr_scheduler.LambdaLR` with the appropriate schedule.
    """
    if num_decay_steps is None:
        num_decay_steps = 2 * num_warmup_steps

    lr_lambda = partial(_get_stable_decay_schedule_with_warmup_lr_lambda, num_warmup_steps=num_warmup_steps, num_decay_steps=num_decay_steps, num_training_steps=num_training_steps)
    return LambdaLR(optimizer, lr_lambda, last_epoch=last_epoch)

def _get_stable_decay_schedule_with_warmup_lr_lambda(current_step: int, num_warmup_steps: int, num_decay_steps: int, num_training_steps: int):
    if current_step < num_warmup_steps:
        return float(current_step) / float(max(1, num_warmup_steps))
    elif current_step < num_training_steps - num_decay_steps:
        return 1.0
    return float(num_training_steps - current_step) / float(max(1, num_decay_steps))


