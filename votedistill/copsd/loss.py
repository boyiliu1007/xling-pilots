"""COPSD distillation loss."""
# Copied verbatim from OPSDTrainer.generalized_jsd_loss in https://github.com/cisnlp/COPSD (commit 4f4410730e9b),
# which extends TRL's GKDTrainer.generalized_jsd_loss (https://github.com/huggingface/trl, Apache-2.0) with top_k and
# token_clip. The COPSD repository does not declare a license; all credit for this function belongs to its authors.

def generalized_jsd_loss(student_logits, teacher_logits, labels=None, beta=0.5, temperature=1.0, reduction="batchmean",
                         logits_are_probs=False, top_k=None, token_clip=None):
    import torch
    import torch.nn.functional as F
    if logits_are_probs:
        student_log_probs = torch.log(student_logits.clamp_min(1e-8))
        teacher_log_probs = torch.log(teacher_logits.clamp_min(1e-8))
    else:
        student_logits = student_logits / temperature
        teacher_logits = teacher_logits / temperature
        if top_k is not None and top_k > 0:
            _, top_k_indices = torch.topk(teacher_logits, k=top_k, dim=-1)
            student_logits = torch.gather(student_logits, dim=-1, index=top_k_indices)
            teacher_logits = torch.gather(teacher_logits, dim=-1, index=top_k_indices)
        student_log_probs = F.log_softmax(student_logits, dim=-1)
        teacher_log_probs = F.log_softmax(teacher_logits, dim=-1)
    if beta == 0:
        jsd = F.kl_div(student_log_probs, teacher_log_probs, reduction="none", log_target=True)
    elif beta == 1:
        jsd = F.kl_div(teacher_log_probs, student_log_probs, reduction="none", log_target=True)
    else:
        beta = torch.tensor(beta, dtype=student_log_probs.dtype, device=student_log_probs.device)
        mixture_log_probs = torch.logsumexp(
            torch.stack([student_log_probs + torch.log1p(-beta), teacher_log_probs + torch.log(beta)]), dim=0)
        kl_teacher = F.kl_div(mixture_log_probs, teacher_log_probs, reduction="none", log_target=True)
        kl_student = F.kl_div(mixture_log_probs, student_log_probs, reduction="none", log_target=True)
        jsd = beta * kl_teacher + (1 - beta) * kl_student
    if token_clip is not None:
        jsd = jsd.clamp(max=token_clip)
    if labels is not None:
        mask = labels != -100
        jsd = jsd[mask]
    if reduction == "batchmean":
        return jsd.sum() / mask.sum() if labels is not None else jsd.sum() / jsd.size(0)
    elif reduction == "sum":
        return jsd.sum()
    elif reduction == "mean":
        return jsd.mean()
    return jsd
