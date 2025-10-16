# NN Coding

## MLP & Parallelism

### Question 1: Implement a MLP and Train on MNIST

**Description:**

Your task is to implement a simple Multi-Layer Perceptron (MLP) and train it on the MNIST dataset. The model should have the following structure:

- An input layer of size 784 (flattened 28x28 images).
- At least one hidden layer (e.g., 128 neurons) with ReLU activation.
- An output layer of size 10 (for digit classification).

**Your implementation should:**

- Implement the MLP model in PyTorch.
- Train the model using cross-entropy loss and an optimizer of your choice.
- Evaluate the model on the test set and report accuracy.

**Constraints:**

- Use PyTorch for implementation.
- Train the model using a single GPU.

```python
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

num_epochs = 5
batch_size = 128
lr = 1e-4
train_data = datasets.MNIST("./", train=True, transform=transforms.ToTensor(), download=True)
test_data = datasets.MNIST("./", train=False, transform=transforms.ToTensor(), download=True)

train_loader = DataLoader(train_data, batch_size)
eval_loader = DataLoader(test_data, batch_size)

def compute_accuracy(model, dataloader):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.cuda(), labels.cuda()
            # images = images.view(images.shape[0],-1)
            outputs = model(images)  # NOTE: you may need to flatten the input
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    return accuracy

input_size = 784
num_classes = 10
hidden_size = 256

class MLP(nn.Module):
  def __init__(self, input_size, hidden_size, num_classes):
    super(MLP, self).__init__()
    self.fc1 = nn.Linear(input_size, hidden_size)
    self.fc2 = nn.Linear(hidden_size, num_classes)
    self.relu = nn.ReLU()

  def forward(self, x):
    # x: [batch_size, num_channel, 28, 28]
    x = x.view(-1, input_size)
    out = self.fc1(x)
    out = self.relu(out)
    out = self.fc2(out)
    return out

model = MLP(input_size, hidden_size, num_classes).to(device)

loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr)

# Training loop
for epoch in range(num_epochs):
  model.train()
  total_loss = 0
  for image, label in train_loader:
    image = image.to(device)
    label = label.to(device)
    # print(label.size())

    output = model(image)
    loss = loss_fn(output, label)

    # backward
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    total_loss += loss.item()

  print(f"[Epoch {epoch} done]. Loss={total_loss}")

compute_accuracy(model, eval_loader)
```

### Question 2: Convert the MLP to Use Distributed Data Parallel (DDP)

**Description:**

Now, you need to parallelize the training of your MLP using Distributed Data Parallel (DDP) across two GPUs. Modify your implementation from Question 1 to:

- Initialize the PyTorch distributed environment (`torch.distributed.init_process_group`).
- Modify the model to use `torch.nn.parallel.DistributedDataParallel` (DDP).
- Use `DistributedSampler` to ensure data is evenly split across multiple GPUs.
- Ensure proper synchronization and cleanup of the distributed process.
- Train the model using DDP on 2 GPUs and evaluate its accuracy.

**Constraints:**

- Use Distributed Data Parallel (DDP).
- Don't use `torchrun`

```python
# mnist_ddp.py

import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from torchvision import datasets, transforms


class MLP(nn.Module):
  def __init__(self, input_size, hidden_size, num_classes):
    super(MLP, self).__init__()
    self.fc1 = nn.Linear(input_size, hidden_size)
    self.fc2 = nn.Linear(hidden_size, num_classes)
    self.relu = nn.ReLU()

  def forward(self, x):
    # x: [batch_size, num_channel, 28, 28]
    x = x.view(-1, input_size)
    out = self.fc1(x)
    out = self.relu(out)
    out = self.fc2(out)
    return out


def train(rank, world_size):
  dist.init_process_group(backend='nccl', rank=rank, world_size=world_size)

  torch.cuda.set_device()

  # config
  num_epochs = 5
  batch_size = 128
  lr = 1e-4
  input_size = 784
  num_classes = 10
  hidden_size = 256

  # datasets
  train_data = datasets.MNIST("./", train=True, transform=transforms.ToTensor(), download=True)
  test_data = datasets.MNIST("./", train=False, transform=transforms.ToTensor(), download=True)

  train_sampler = DistributedSampler(train_data, num_replicas=world_size, rank=rank, shuffle=True)
  test_sampler = DistributedSampler(test_data, num_replicas=world_size, rank=rank, shuffle=False)

  train_dataloader = DataLoader(train_data, batch_size=batch_size, sampler=train_sampler)
  test_dataloader = DataLoader(test_data, batch_size=batch_size, sampler=test_sampler)

  # Model
  device = torch.device(f"cuda:{rank}")
  model = MLP(input_size, hidden_size, num_classes).to(device)
  model = DDP(model, device_ids=[rank])
  loss_fn = nn.CrossEntropyLoss()
  optimizer = optim.Adam(model.parameters(), lr)

  # Training loop
  for epoch in range(num_epochs):
    model.train()
    for image, label in train_dataloader:
      image = image.to(device)
      label = label.to(device)
      # print(label.size())

      output = model(image)
      loss = loss_fn(output, label)

      # backward
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()

  # Eval
  if rank == 0:
    pass


def main():
  world_size = 2
  mp.spawn(train,
      args=(world_size,),
      nprocs=world_size,
      join=True)

if __name__ == "__main__":
  main()

'''
!python -m torch.distributed.launch mnist_ddp.py
'''
```

## Tensor Parallelism

**Description:**

In deep learning, when working with large tensors, tensor parallelism can be used to shard tensors across multiple devices to efficiently utilize computational resources. Suppose you have 2 GPUs and need to perform a forward pass through a 2-layer MLP on an input tensor of shape (2, 2).

- Devise a sharding strategy that minimizes the number of synchronizations across devices.
- Implement this strategy using `torch.distributed.dist`. You can assume this script will be launched by `torchrun --nproc-per-node 2`