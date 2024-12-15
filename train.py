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

EPOCHS=15

# Check for Metal device
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
print(f"Using device: {device}")

torch.manual_seed(1)

train_transform = transforms.Compose(
                    [
                    # transforms.RandomAffine(degrees=7, translate=(0.1,0.1)),
                    #transforms.RandomRotation(5),
                    # transforms.ColorJitter(brightness=0.10, contrast=0.1, saturation=0.10, hue=0.1),
                    transforms.Resize((28, 28)),
                    transforms.RandomRotation((-7.0, 7.0), fill=(1,)), 
                    transforms.ToTensor(),
                    transforms.Normalize((0.5,), (0.5,))
                    ])

test_transform = transforms.Compose(
                    [
                    transforms.ToTensor(),
                    transforms.Normalize((0.5,), (0.5,))
                    ])

def train_model(model, device, train_loader, optimizer, scheduler, criterion, epoch):
    model.train()
    pbar = tqdm(train_loader)
    train_loss = 0
    correct = 0
    processed = 0
    
    for batch_idx, (data, target) in enumerate(pbar):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        train_loss += loss.item()
        
        # Calculate Training Accuracy
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()
        processed += len(data)
        
        loss.backward()
        optimizer.step()
        scheduler.step()
        
        pbar.set_description(desc= f'Training Model: Loss={loss.item():.4f} Batch={batch_idx} Accuracy={100*correct/processed:.2f}%')
    
    # Print epoch summary
    train_loss /= len(train_loader.dataset)
    print('Training: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)'.format(
        train_loss, correct, len(train_loader.dataset),
        100. * correct / len(train_loader.dataset)))
    
    return correct / len(train_loader.dataset)

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
    BATCH_SIZE=16
    train_dataset = datasets.MNIST('../data', train=True, download=True, transform=train_transform)
    test_dataset = datasets.MNIST('../data', train=False, transform=test_transform)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True)

    return train_loader, test_loader

def main():
    model = Net().to(device)

    # Move summary to CPU since torchsummary has issues with MPS
    summary(model.to('cpu'), input_size=(1, 28, 28))

    # Move model back to MPS device
    model = model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    train_loader, test_loader = load_data()


    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer,max_lr=0.01,epochs=EPOCHS,steps_per_epoch=len(train_loader))


    epoc_id = 0
    best_acc_test = 0
    max_epochs =  EPOCHS + 1
    for epoch in range(1, max_epochs):
        print(f'Epoch: {epoch}')

        train_acc = train_model(model, device, train_loader, optimizer, scheduler, criterion, epoch)
        test_acc = test_model(model, device, test_loader, criterion, epoch)
        
        if test_acc > best_acc_test:
            best_acc_test = test_acc
            epoc_id = epoch
            # Save best model
            torch.save(model.state_dict(), 'model.pth')

    print(f'Best test Accuracy Achieved: {best_acc_test * 100:.2f}%, Epoch: {epoc_id}')

if __name__ == "__main__":
    main() 