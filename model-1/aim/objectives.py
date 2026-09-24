"""Separate mathematical objectives. No implicit weighted mixture."""
import torch
from torch.nn import functional as F


def dpo_loss(policy_chosen, policy_rejected, reference_chosen, reference_rejected, beta=0.1):
    margin = (policy_chosen - policy_rejected) - (reference_chosen - reference_rejected)
    return -F.logsigmoid(beta * margin).mean()


def reinforce_loss(logits, reference_logits, rewards, action, kl_coefficient=0.02):
    logp = logits.log_softmax(-1)
    p = logp.exp()
    logref = reference_logits.detach().log_softmax(-1)
    baseline = (p.detach() * rewards).sum()
    advantage = (rewards[action] - baseline).detach()
    kl = (p * (logp - logref)).sum()
    return -advantage * logp[action] + kl_coefficient * kl, kl


def proper_loss(logits, labels, scoring_rule="log"):
    if scoring_rule == "log":
        return F.binary_cross_entropy_with_logits(logits, labels)
    if scoring_rule == "brier":
        return (logits.sigmoid() - labels).square().mean()
    raise ValueError("Only log and brier are proper-score options")
