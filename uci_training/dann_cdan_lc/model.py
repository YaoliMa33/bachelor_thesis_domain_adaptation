"""Shared feature extractor, classifier, and domain discriminator."""
import random

import numpy as np
import torch
from torch import nn


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


class GradientReversal(torch.autograd.Function):
    @staticmethod
    def forward(ctx, features, coefficient):
        ctx.coefficient = coefficient
        return features.view_as(features)

    @staticmethod
    def backward(ctx, gradient):
        return -ctx.coefficient * gradient, None


class AdaptationModel(nn.Module):
    def __init__(self, method, config):
        super().__init__()
        self.extractor = nn.Sequential(nn.Linear(config.input_dim, config.hidden_dim),
                                       nn.BatchNorm1d(config.hidden_dim), nn.ReLU())
        self.classifier = nn.Linear(config.hidden_dim, config.classes)
        domain_input = config.hidden_dim * config.classes if method.startswith("CDAN") else config.hidden_dim
        self.domain = nn.Sequential(nn.Linear(domain_input, config.domain_hidden), nn.ReLU(),
                                    nn.Linear(config.domain_hidden, config.domain_hidden), nn.ReLU(),
                                    nn.Linear(config.domain_hidden, 1))

    def forward(self, x):
        return self.classifier(self.extractor(x))


@torch.no_grad()
def project_domain(model, cap):
    for layer in model.domain:
        if isinstance(layer, nn.Linear):
            sigma = (torch.linalg.vector_norm(layer.weight) if layer.weight.shape[0] == 1
                     else torch.linalg.matrix_norm(layer.weight, ord=2))
            layer.weight.mul_((cap / sigma).clamp(max=1.0))


def build_model(method, config, seed):
    set_seed(seed)
    model = AdaptationModel(method, config)
    # Reset after construction so the shared extractor starts identically across methods.
    set_seed(seed)
    for layer in model.modules():
        if isinstance(layer, nn.Linear):
            nn.init.xavier_normal_(layer.weight)
            nn.init.zeros_(layer.bias)
    project_domain(model, config.spectral_cap)
    torch.manual_seed(seed + 10000)
    return model
