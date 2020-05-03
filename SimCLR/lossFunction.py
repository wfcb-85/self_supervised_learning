import torch

# Translated to Pytorch from
# https://github.com/google-research/simclr/blob/master/objective.py

def simCLRLossFunction(hidden1, hidden2, criterion):

    hidden1Shape = hidden1.shape
    LARGE_NUM = 1e9
    # First we normalize
    hidden1Norm  = torch.norm(hidden1, p=2, dim=1, keepdim=True)
    hidden2Norm = torch.norm(hidden2, p=2, dim=1, keepdim=True)

    hidden1 = hidden1 / hidden1Norm
    hidden2 = hidden2 / hidden2Norm

    maskTorch = torch.eye(hidden1Shape[0], hidden1Shape[0]).cuda()

    logits_aa_torch = torch.matmul(hidden1, hidden1.T).cuda()
    logits_bb_torch = torch.matmul(hidden2, hidden2.T).cuda()

    logits_aa_torch = logits_aa_torch - maskTorch * LARGE_NUM
    logits_bb_torch = logits_bb_torch - maskTorch * LARGE_NUM

    logits_ab_torch = torch.matmul(hidden1, hidden2.T).cuda()
    logits_ba_torch = torch.matmul(hidden2, hidden1.T).cuda()

    concat_a = torch.cat([logits_ab_torch, logits_aa_torch], 1)
    concat_b = torch.cat([logits_ba_torch, logits_bb_torch], 1)

    torch_labels = torch.tensor(range(0, hidden1Shape[0])).cuda()

    # loss = torch.nn.CrossEntropyLoss()

    losses_a_torch = criterion(concat_a, torch_labels)
    losses_b_torch = criterion(concat_b, torch_labels)

    totalLoss = losses_a_torch + losses_b_torch

    return totalLoss