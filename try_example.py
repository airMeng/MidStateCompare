import random
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader, Dataset
from compare_dict import compare_state
from torch.autograd import Variable
from save_state import save_state

class TwoLayerNet(torch.nn.Module):
    def __init__(self, D_in, H, D_out):
        """
        In the constructor we instantiate two nn.Linear modules and assign them as
        member variables.
        """
        super(TwoLayerNet, self).__init__()
        self.linear1 = torch.nn.Linear(D_in, H)
        self.linear2 = torch.nn.Linear(H, D_out)

    def forward(self, x):
        """
        In the forward function we accept a Tensor of input data and we must return
        a Tensor of output data. We can use Modules defined in the constructor as
        well as arbitrary operators on Tensors.
        """
        h_relu = self.linear1(x).clamp(min=0)
        y_pred = self.linear2(h_relu)
        return y_pred


class DynamicNet(torch.nn.Module):
    def __init__(self, D_in, H, D_out):
        """
        In the constructor we construct three nn.Linear instances that we will use
        in the forward pass.
        """
        super(DynamicNet, self).__init__()
        self.input_linear = torch.nn.Linear(D_in, H)
        self.middle_linear = torch.nn.Linear(H, H)
        self.output_linear = torch.nn.Linear(H, D_out)

    def forward(self, x):
        h_relu = self.input_linear(x).clamp(min=0)
        for _ in range(random.randint(0, 3)):
            h_relu = self.middle_linear(h_relu).clamp(min=0)
        y_pred = self.output_linear(h_relu)
        return y_pred


class Net(torch.nn.Module):
    def __init__(self, m1,m2):
        """
        In the constructor we instantiate two nn.Linear modules and assign them as
        member variables.
        """
        super(Net, self).__init__()
        self.m1 = m1
        self.m2 = m2

    def forward(self, x):
        """
        In the forward function we accept a Tensor of input data and we must return
        a Tensor of output data. We can use Modules defined in the constructor as
        well as arbitrary operators on Tensors.
        """

        return self.m2(self.m1(x))


seed=123
random.seed(seed)
torch.manual_seed(seed)
cudnn.deterministic = True
# N is batch size; D_in is input dimension;
# H is hidden dimension; D_out is output dimension.
N, D_in, H, D_out = 64, 1000, 100, 10

# Create random Tensors to hold inputs and outputs
X = torch.randn(N, D_in)
Y = torch.randn(N, D_out)


class TestDataset(Dataset):
    def __len__(self):
        return N

    def __getitem__(self, index):
        return X[index, :], Y[index, :]


dataset = TestDataset()
loader = DataLoader(dataset, batch_size=8, shuffle=False, num_workers=1, drop_last=True)


# def export_weight(m):
def init_weights(m):
    print(m)

    m.weight.data.fill_(1.0)
    print(m.weight)


# Construct our model by instantiating the class defined above
model = Net(DynamicNet(D_in,H,D_in), TwoLayerNet(D_in,H,D_out))
# model=model.cuda()
model.train()
criterion = torch.nn.MSELoss(reduction='sum')
optimizer = torch.optim.SGD(model.parameters(), lr=1e-4, momentum=0.9)
for t in range(5):
    # Forward pass: Compute predicted y by passing x to the model
    # torch.manual_seed(seed + t)
    for iter,batch in enumerate(loader):
        x,y=batch
  #      x=x.cuda()
   #     y=y.cuda()
        with save_state(model=model,epoch=t,iter=iter):
            y_pred = model(x)
        # Compute and print loss
        loss = criterion(y_pred, y)
        # state1=model.state_dict()
        optimizer.zero_grad()

        # state2 = model.state_dict()
        loss.backward()
        # state3 = model.state_dict()
        optimizer.step()
        # state4 = model.state_dict()


