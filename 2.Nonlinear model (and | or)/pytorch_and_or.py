import torch
from sympy.printing.codeprinter import requires
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader, TensorDataset

data = [[0., 0.], [0., 1.], [1., 0.], [1., 1.]]
labels = [0., 0., 0., 1.]

X = torch.tensor(data, requires_grad=True).float()
y = torch.tensor([labels], requires_grad=True).mT.float()


dataset = TensorDataset(X, y)

batch_size = len(X)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)


device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
print(f"Using {device} device")

# Define model
class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(2, 1, bias = True),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.flatten(x)
        return self.linear_relu_stack(x)

model = NeuralNetwork().to(device)
print(model)

loss_fn = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

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

epochs = 5000


for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train(dataloader, model, loss_fn, optimizer)
    # test(test_dataloader, model, loss_fn)
print("Done!")



input_value = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=torch.float32).to(device)
model.eval()

with torch.no_grad():
    prediction = model(input_value)

print(prediction[0].item(), round(prediction[0].item()))
print(prediction[1].item(), round(prediction[1].item()))
print(prediction[2].item(), round(prediction[2].item()))
print(prediction[3].item(), round(prediction[3].item()))
