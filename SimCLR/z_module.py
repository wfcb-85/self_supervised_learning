import torch

class z_module(torch.nn.Module):

    def __init__(self, resnetArchitecture, H=2048, D_out=4096):
        super(z_module, self).__init__()

        if resnetArchitecture == 18:
            D_in = 512

        elif resnetArchitecture == 34:
            D_in = 512

        elif resnetArchitecture == 50:
            D_in = 2048

        elif resnetArchitecture == 101:
            D_in = 2048

        elif resnetArchitecture == 152:
            D_in = 2048

        self.linear1 = torch.nn.Linear(D_in, H)
        self.linear2 = torch.nn.Linear(H, D_out)
        self.relu = torch.nn.ReLU(inplace=True)

    def forward(self, x):

        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)

        return x

