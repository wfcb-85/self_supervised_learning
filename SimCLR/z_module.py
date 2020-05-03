import torch

class z_module(torch.nn.Module):

    def __init__(self, D_in, H, D_out):
        super(z_module, self).__init__()

        self.linear1 = torch.nn.Linear(D_in, H)
        self.linear2 = torch.nn.Linear(H, D_out)
        self.relu = torch.nn.ReLU(inplace=True)

    def forward(self, x):

        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)

        return x

