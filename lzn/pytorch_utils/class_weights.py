import torch


def get_class_weights(label, num_classes):
    label_aug = torch.concat(
        [label, torch.arange(0, num_classes).to(label.device).type_as(label)],
        dim=0,
    )
    _, counts = torch.unique(label_aug, sorted=True, return_counts=True)
    counts = counts - 1
    torch.distributed.all_reduce(counts)
    torch.distributed.barrier()
    counts = counts.float()
    weights = counts / counts.sum()
    return weights
