import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np
from model import Net
from torchsummary import summary
from tqdm import tqdm
import torch.nn.functional as F
from torch.optim.lr_scheduler import OneCycleLR

torch.manual_seed(42)

train_transform = transforms.Compose(
                    [
                    transforms.RandomAffine(degrees=7, translate=(0.1,0.1)),
                    transforms.ColorJitter(brightness=0.2, contrast=0.2),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5,), (0.5,)),
                    ])

test_transform = transforms.Compose(
                    [
                    transforms.ToTensor(),
                    transforms.Normalize((0.5,), (0.5,)),
                    ])

def train_model(model, device, train_loader, optimizer, scheduler, criterion, epoch):
    model.train()
    pbar = tqdm(train_loader)
    for batch_idx, (data, target) in enumerate(pbar):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        scheduler.step()
        pbar.set_description(desc= f'Training Model: loss={loss.item():.4f} batch_id={batch_idx}')

def test_model(model, device, test_loader, criterion, epoch):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()  
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('Testing Model: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100.0 * correct / len(test_loader.dataset)))
    return correct / len(test_loader.dataset)
        
def load_data():
    batch_size = 24
    train_dataset = datasets.MNIST('../data', train=True, download=True, transform=train_transform)
    test_dataset = datasets.MNIST('../data', train=False, transform=test_transform)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    return train_loader, test_loader

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Net().to(device)

    summary(model, input_size=(1, 28, 28))
    
    criterion = nn.CrossEntropyLoss()
    train_loader, test_loader = load_data()

    optimizer = optim.Adam(model.parameters(), lr=0.01)
    scheduler = OneCycleLR(
        optimizer,
        max_lr=0.01,
        epochs=20,
        steps_per_epoch=len(train_loader),
        pct_start=0.2,
        div_factor=10,
        final_div_factor=100
    )

    epoc_id = 0
    best_acc_test = 0
    for epoch in range(1, 21):
        print(f'Epoch: {epoch}')

        train_model(model, device, train_loader, optimizer, scheduler,criterion, epoch)
        acc_test = test_model(model, device, test_loader,criterion, epoch)
        
        if acc_test > best_acc_test:
            best_acc_test = acc_test
            epoc_id = epoch
            #Save best model
            torch.save(model.state_dict(), 'model.pth')

    print(f'Best test Accuracy Achieved: {best_acc_test * 100:.2f}%, Epoch: {epoc_id}')

if __name__ == "__main__":
    main() 