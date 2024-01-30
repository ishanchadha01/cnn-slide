from sketch_conv import SketchConv2d
from models import MNIST, AlshCNN
import torch
from torchvision import transforms, datasets
import torch.nn.functional as F
import torch.optim as optim
import argparse
from tqdm import tqdm


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=100)
    parser.add_argument("--lr", type=float, default=0.2)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--gamma", type=float, default=0.7, help="Step for learning rate")
    args = parser.parse_args()

    device = torch.device("cpu")
    # model = MNIST().to(device)
    model = AlshCNN().to(device)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    train_data = datasets.MNIST("../data", train=True, transform=transform, download=False)
    test_data = datasets.MNIST("../data", train=False, transform=transform, download=False)
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=args.batch_size)

    optimizer = optim.Adadelta(params=model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=1, gamma=args.gamma)

    print("Training")
    for epoch in tqdm(range(1, args.epochs+1)):
        # train
        model.train()
        for batch_idx, (data, target) in tqdm(enumerate(train_loader)):
            data = data.to(device)
            target = target.to(device)
            optimizer.zero_grad()
            out = model(data)
            loss = F.nll_loss(out, target)
            loss.backward()
            optimizer.step()
            if(batch_idx>20): break # we dont actually need to train for very long with this model


        # test
        print("Testing")
        model.eval()
        test_loss = 0
        pred = None
        correct = 0
        with torch.no_grad():
            for data, target in tqdm(test_loader):
                data = data.to(device)
                target = target.to(device)
                out = model(data)
                test_loss += F.nll_loss(out, target, reduction="sum").item()
                pred = out.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
        print(f"Test Loss: {test_loss / len(test_data)}")
        print(f"Test Accuracy: {correct / len(test_data)}")
        scheduler.step()