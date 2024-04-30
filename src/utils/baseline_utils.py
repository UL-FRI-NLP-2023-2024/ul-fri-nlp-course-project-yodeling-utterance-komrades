import torch
import torch.nn as nn
from typing import Dict

def get_criterion(config: Dict[str, any]) -> torch.nn.modules.loss._Loss:
    """
    Get the criterion based on the configuration.

    Args:
        config (dict): The configuration dictionary.

    Returns:
        torch.nn.modules.loss._Loss: The criterion to be used for training the model.

    Raises:
        ValueError: If the loss function is not specified or if the loss function is invalid.
    """
    if config['loss_function'] == 'bce_with_logits':
        criterion = nn.BCEWithLogitsLoss()
    elif config['loss_function'] == 'cross_entropy':
        criterion = nn.CrossEntropyLoss()
    elif config['loss_function'] == 'multi_label_soft_margin':
        criterion = nn.MultiLabelSoftMarginLoss()
    elif config['loss_function'] == 'multi_label_margin':
        criterion = nn.MultiLabelMarginLoss()
    else:
        raise ValueError('Invalid loss function: {}. Accepted values are: bce_with_logits, cross_entropy, multi_label_soft_margin, multi_label_margin'.format(config['loss_function']))
    return criterion
    

def get_optimizer(config: Dict[str, any], classifier: nn.Module) -> torch.optim.Optimizer:
    """
    Get the optimizer based on the configuration.

    Args:
        config (dict): The configuration dictionary.
        classifier (nn.Module): The classifier model.

    Returns:
        torch.optim.Optimizer: The optimizer to be used for training the model.

    Raises:
        ValueError: If the learning rate is not specified for the Adam optimizer.
    """
    if config['optimizer'] == 'adam':
        if 'learning_rate' not in config:
            raise ValueError('Learning rate not specified for Adam optimizer')
        optimizer = torch.optim.Adam(classifier.parameters(), lr=config['learning_rate'])
    elif config['optimizer'] == 'sgd':
        if 'learning_rate' not in config:
            raise ValueError('Learning rate not specified for SGD optimizer')
        optimizer = torch.optim.SGD(classifier.parameters(), lr=config['learning_rate'])
    else:
        raise ValueError('Invalid optimizer: {}. Accepted values are: adam, sgd'.format(config['optimizer']))
    return optimizer


def get_scheduler(config: Dict[str, any], optimizer: torch.optim.Optimizer) -> torch.optim.lr_scheduler._LRScheduler:
    """
    Get the scheduler based on the configuration.

    Args:
        config (dict): The configuration dictionary.
        optimizer (torch.optim.Optimizer): The optimizer.

    Returns:
        torch.optim.lr_scheduler._LRScheduler: The scheduler to be used for training the model.

    Raises:
        ValueError: If the scheduler gamma is not specified or if the scheduler is invalid.
    """
    if config['scheduler'] != 'none':
        if 'scheduler_gamma' not in config:
            raise ValueError('Scheduler gamma not specified')
        if config['scheduler'] == 'linear':
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=config['scheduler_step'], gamma=config['scheduler_gamma'])
        elif config['scheduler'] == 'exponential':
            scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=config['scheduler_gamma'])
        else:
            raise ValueError('Invalid scheduler: {}. Accepted values are: none, linear'.format(config['scheduler']))
    else:
        scheduler = None
    return scheduler