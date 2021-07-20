import torch.nn as nn

class Criterion:
    CE = nn.CrossEntropyLoss()
    MSE = nn.MSELoss()
    NLL = nn.NLLLoss()
    HE = nn.HingeEmbeddingLoss()
    KL = nn.KLDivLoss()
    BCE = nn.BCELoss()