import torch
import torch.nn as nn

# Define a simple neural network
class SimpleNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(4, 8)   # 4 inputs → 8 neurons
        self.layer2 = nn.Linear(8, 4)   # 8 neurons → 4 neurons
        self.layer3 = nn.Linear(4, 1)   # 4 neurons → 1 output
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.layer1(x))
        x = self.relu(self.layer2(x))
        x = self.layer3(x)
        return x

model = SimpleNN()
print(model)

# Create fake input data — 5 samples, 4 features each
input_data = torch.randn(5, 4)
print("Input shape:", input_data.shape)

# Forward pass
output = model(input_data)
print("Output shape:", output.shape)
print("Output:", output)

# Loss function and optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# Fake target values
targets = torch.randn(5, 1)

# One training step
optimizer.zero_grad()        # reset gradients
output = model(input_data)   # forward pass
loss = criterion(output, targets)  # calculate loss
loss.backward()              # calculate gradients
optimizer.step()             # update weights

print(f"Loss: {loss.item():.4f}")

# Full training loop
losses = []

for epoch in range(100):
    optimizer.zero_grad()
    output = model(input_data)
    loss = criterion(output, targets)
    loss.backward()
    optimizer.step()
    losses.append(loss.item())

print(f"Starting loss: {losses[0]:.4f}")
print(f"Final loss: {losses[-1]:.4f}")
import matplotlib.pyplot as plt

plt.plot(losses)
plt.title('Training Loss over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.savefig('training_loss.png')
print("Chart saved!")