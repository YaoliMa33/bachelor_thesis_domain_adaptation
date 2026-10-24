"""UCI entropy conditioning; shared modules come from ../dann_cdan_lc.

Run this file directly to reuse the shared experiment entry point with +E training.
"""
import sys
from pathlib import Path

SHARED = Path(__file__).resolve().parent.parent / "dann_cdan_lc"
sys.path.insert(0, str(SHARED))
import torch
from torch.nn import functional as F

from model import GradientReversal
from train import latent_center_loss, train as shared_train


def entropy_domain_loss(probabilities, domain_logits, n_source, coefficient):
    # Original thuml/CDAN pytorch/loss.py semantics, with stable logits BCE.
    entropy = -(probabilities * torch.log(probabilities + 1e-5)).sum(1)
    entropy = GradientReversal.apply(entropy, coefficient)
    weights = 1.0 + torch.exp(-entropy)
    sw, tw = weights[:n_source], weights[n_source:]
    weights = torch.cat([sw / sw.sum().detach(), tw / tw.sum().detach()])
    labels = torch.cat([torch.ones_like(sw), torch.zeros_like(tw)])
    terms = F.binary_cross_entropy_with_logits(domain_logits, labels, reduction='none')
    return (weights * terms).sum() / weights.sum().detach()


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
    probabilities = torch.cat([source_logits, target_logits]).softmax(1)
    domain_logits = torch.cat([source_domain_logits, target_domain_logits])
    domain = entropy_domain_loss(probabilities, domain_logits, len(source_x), coefficient)
    lc = latent_center_loss(source_features, source_y, target_features, target_logits,
                            config.classes, config.epsilon)
    weight = config.lc_weight if method.endswith("LC") else 0.0
    return dict(total=classification + config.domain_weight * domain + weight * lc,
                classification=classification, domain=domain, lc=lc)


def train(method, source_x, source_y, target_x, config, seed):
    return shared_train(method, source_x, source_y, target_x, config, seed,
                        loss_fn=losses)


def main():
    """Reuse the original CLI, data pipeline, evaluation and result saving."""
    import main as shared_main

    shared_main.train = train
    # Separate output location prevents overwriting the baseline results by default.
    if not any(a == "--output" or a.startswith("--output=") for a in sys.argv[1:]):
        sys.argv.extend(["--output", str(Path(__file__).resolve().parent / "results")])
    if not any(a == "--uci-protocol" or a.startswith("--uci-protocol=") for a in sys.argv[1:]):
        sys.argv.extend(["--uci-protocol", "long"])
    print("Entropy conditioning enabled (+E); shared UCI modules: " + str(SHARED), flush=True)
    shared_main.main(entropy_conditioning=True)


if __name__ == "__main__":
    main()
