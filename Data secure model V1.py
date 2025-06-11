# supprt libraries for the AI model
import torch
import torch.nn as nn
import torch.optim as optim
import syft as sy  # PySyft for federated learning

#definition of a simple AI Model
class PrivacyPreservingAI(nn.Module):
    def __init__(self):
        super(PrivacyPreservingAI, self).__init__()
        self.fc1 = nn.Linear(2, 4)
        self.fc2 = nn.Linear(4, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        return x
#Decentralized data source(virtual workers)
# Creating  a federated learning environment with virtual workers
hook = sy.TorchHook(torch)  # Hook PyTorch for federated learning
worker1 = sy.VirtualWorker(hook, id="worker1")
worker2 = sy.VirtualWorker(hook, id="worker2")

# Sample training data (each worker gets a portion)
data = torch.tensor([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6], [0.7, 0.8]])
labels = torch.tensor([[0.0], [1.0], [1.0], [0.0]])

# Split data between workers
data_worker1 = data[:2].send(worker1)
labels_worker1 = labels[:2].send(worker1)

data_worker2 = data[2:].send(worker2)
labels_worker2 = labels[2:].send(worker2)

# Initialize model and optimizer
model = PrivacyPreservingAI()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# Model training  on the  decentralized data
for epoch in range(5):
    for worker, data, labels in [(worker1, data_worker1, labels_worker1), (worker2, data_worker2, labels_worker2)]:
        model.send(worker)  # Send model to worker
        optimizer.zero_grad()
        output = model(data)
        loss = nn.BCELoss()(output, labels)
        loss.backward()
        optimizer.step()
        model.get()  # Get updated model back from worker

    print(f"Epoch {epoch+1}, Loss: {loss.item()}")

#model test without data retension
test_data = torch.tensor([[0.25, 0.35]])
model.eval()
prediction = model(test_data)
print("Prediction:", prediction.item())

