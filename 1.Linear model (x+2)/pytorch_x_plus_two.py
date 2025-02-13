import torch
from sympy.printing.codeprinter import requires
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader, TensorDataset

data = [1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 14., 15., 16., 17., 19., 20., 22., 23.]
labels = [3., 4., 5., 6., 7., 8., 9., 10., 11., 12., 16., 17., 18., 19., 21., 22., 24., 25.]




X = torch.tensor([data], requires_grad=True).mT.float()
y = torch.tensor([labels], requires_grad=True).mT.float()


# X = X.float()
# y = y.float()

dataset = TensorDataset(X, y)

batch_size = len(X)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)


# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print(f"Using {device} device")


device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
print(f"Using {device} device")

# Define model
class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(1, 1, bias = True)
        )

    def forward(self, x):
        x = self.flatten(x)
        return self.linear_relu_stack(x)

model = NeuralNetwork().to(device)
print(model)

loss_fn = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.001)

def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)

        loss = loss_fn(pred, y)


        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()


        loss, current = loss.item(), (batch + 1) * len(X)
        print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

epochs = 100
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train(dataloader, model, loss_fn, optimizer)
    # test(test_dataloader, model, loss_fn)
print("Done!")



input_value = torch.tensor([[12.]], dtype=torch.float32).to(device)
model.eval()

with torch.no_grad():
    prediction = model(input_value)

print(f"{prediction.item()}")