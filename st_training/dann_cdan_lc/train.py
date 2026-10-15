"""One training implementation for both datasets and all four methods."""
import numpy as np
import torch
from torch.nn import functional as F

from model import GradientReversal, build_model, project_domain
from utils import Stratified, continuous_target, make_loaders, repeat


def latent_center_loss(source_features, source_labels, target_features, target_logits,
                       classes, epsilon):
    probabilities = F.softmax(target_logits, dim=1).detach()
    weights = F.one_hot(source_labels, num_classes=classes).to(source_features.dtype)
    source_mass = weights.sum(0)
    source_centers = weights.T @ source_features / source_mass.clamp_min(epsilon)[:, None]
    target_mass = probabilities.sum(0)
    target_centers = probabilities.T @ target_features / target_mass.clamp_min(epsilon)[:, None]
    present = source_mass > 0
    return (source_centers[present] - target_centers[present]).square().mean()


def losses(model, method, source_x, source_y, target_x, coefficient, config):
    features = model.extractor(torch.cat([source_x, target_x]))
    source_features, target_features = features.split([len(source_x), len(target_x)])
    source_logits = model.classifier(source_features)
    target_logits = model.classifier(target_features)
    classification = F.cross_entropy(source_logits, source_y)
    source_domain, target_domain = source_features, target_features
    if method.startswith("CDAN"):
        source_domain = (source_features.unsqueeze(2) * F.softmax(source_logits, dim=1).detach().unsqueeze(1)).reshape(len(source_x), -1)
        target_domain = (target_features.unsqueeze(2) * F.softmax(target_logits, dim=1).detach().unsqueeze(1)).reshape(len(target_x), -1)
    source_domain_logits = model.domain(GradientReversal.apply(source_domain, coefficient)).squeeze(1)
    target_domain_logits = model.domain(GradientReversal.apply(target_domain, coefficient)).squeeze(1)
    domain = 0.5 * (F.binary_cross_entropy_with_logits(source_domain_logits, torch.ones_like(source_domain_logits))
                    + F.binary_cross_entropy_with_logits(target_domain_logits, torch.zeros_like(target_domain_logits)))
    lc = latent_center_loss(source_features, source_y, target_features, target_logits,
                            config.classes, config.epsilon)
    weight = config.lc_weight if method.endswith("LC") else 0.0
    return dict(total=classification + config.domain_weight * domain + weight * lc,
                classification=classification, domain=domain, lc=lc)


def train(method, source_x, source_y, target_x, config, seed):
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    model = build_model(method, config, seed)
    if config.optimizer == "Adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate,
                                     weight_decay=config.weight_decay)
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=config.learning_rate,
                                    momentum=config.momentum, weight_decay=config.weight_decay)
    scheduler = (torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.epochs,
                                                           eta_min=config.eta_min)
                 if config.scheduler == "cosine" else None)
    source_loader, target_loader = make_loaders(source_x, source_y, target_x, config, seed)
    stratified = (Stratified(source_y.numpy(), config.batch_size, seed + 11000)
                  if config.source_sampling == "stratified" else None)
    persistent_target = continuous_target(target_loader) if config.target_sampling == "continuous" else None
    history = []
    for epoch in range(config.epochs):
        model.train()
        source_iterator = repeat(source_loader)
        target_iterator = persistent_target if persistent_target is not None else repeat(target_loader)
        totals = np.zeros(4)
        lr = optimizer.param_groups[0]["lr"]
        for step in range(config.steps_per_epoch):
            if stratified is None:
                xs, ys = next(source_iterator)
            else:
                indices = stratified.take()
                xs, ys = source_x[indices], source_y[indices]
            (xt,) = next(target_iterator)
            ramp = config.grl_ramp_updates or config.epochs * config.steps_per_epoch
            progress = min(epoch * config.steps_per_epoch + step, ramp - 1) / (ramp - 1)
            coefficient = float(2 / (1 + np.exp(-config.gamma * progress)) - 1)
            values = losses(model, method, xs, ys, xt, coefficient, config)
            optimizer.zero_grad()
            values["total"].backward()
            optimizer.step()
            project_domain(model, config.spectral_cap)
            totals += [float(values[k].detach()) for k in ("total", "classification", "domain", "lc")]
        history.append(dict(epoch=epoch + 1,learning_rate=lr,
                            losses=dict(zip(("total", "classification", "domain", "lc"),
                                            (totals / config.steps_per_epoch).tolist()))))
        if scheduler is not None:
            scheduler.step()
    model.eval()
    return model, history
