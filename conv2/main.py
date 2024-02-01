from sketch_conv import SketchConv2d
from models import MNIST, AlshCNN
import torch
from torchvision import transforms, datasets, models
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
import argparse
from tqdm import tqdm
import itertools
from alsh_conv import AlshConv2d, SRPTable


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=100)
    parser.add_argument("--lr", type=float, default=0.2)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--gamma", type=float, default=0.7, help="Step for learning rate")
    args = parser.parse_args()

    device = torch.device("cpu")
    model =  models.alexnet(pretrained=True)  # Assuming CIFAR-10 has 10 classes
    print(model.features)
    # model = AlshCNN().to(device)

    # construct alsh cnn using current weights
    # conv1 = AlshConv2d(in_channels=3, out_channels=64, kernel_size=11, stride=4, padding=2, dilation=1, bias=True, is_first_layer=True, is_last_layer=True, num_hashes=5, num_tables=8, max_bits=16, hash_table=SRPTable)
    # ftr_weight = model.features[0].weight
    # conv1.weight = ftr_weight
    # model.features[0] = conv1

    # conv2 = AlshConv2d(in_channels=64, out_channels=192, kernel_size=5, stride=1, padding=2, dilation=1, bias=True, is_first_layer=True, is_last_layer=True, num_hashes=5, num_tables=8, max_bits=16, hash_table=SRPTable)
    # ftr_weight = model.features[3].weight
    # conv2.weight = ftr_weight
    # model.features[3] = conv2

    # conv3 = AlshConv2d(in_channels=192, out_channels=384, kernel_size=3, stride=1, padding=1, dilation=1, bias=True, is_first_layer=True, is_last_layer=True, num_hashes=5, num_tables=8, max_bits=16, hash_table=SRPTable)
    # ftr_weight = model.features[6].weight
    # conv3.weight = ftr_weight
    # model.features[6] = conv3

    # conv4 = AlshConv2d(in_channels=384, out_channels=256, kernel_size=3, stride=1, padding=1, dilation=1, bias=True, is_first_layer=True, is_last_layer=True, num_hashes=5, num_tables=8, max_bits=16, hash_table=SRPTable)
    # ftr_weight = model.features[8].weight
    # conv4.weight = ftr_weight
    # model.features[8] = conv4

    conv5 = AlshConv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1, dilation=1, bias=True, is_first_layer=True, is_last_layer=True, num_hashes=5, num_tables=8, max_bits=16, hash_table=SRPTable)
    ftr_weight = model.features[10].weight
    conv5.weight = ftr_weight
    model.features[10] = conv5



    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    train_data = datasets.CIFAR10("../data", train=True, transform=transform, download=False)
    test_data = datasets.CIFAR10("../data", train=False, transform=transform, download=False)
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=args.batch_size)

    optimizer = optim.Adadelta(params=model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=1, gamma=args.gamma)

    print("Training")
    for epoch in tqdm(range(1, args.epochs+1)):
        # train
        model.train()
        for batch_idx, (data, target) in tqdm( enumerate(itertools.islice(train_loader, 3)) ):
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
            for data, target in tqdm( itertools.islice(test_loader, 3) ):
                data = data.to(device)
                target = target.to(device)
                out = model(data)
                test_loss += F.nll_loss(out, target, reduction="sum").item()
                pred = out.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
        print(f"Test Loss: {test_loss / len(test_data)}")
        print(f"Test Accuracy: {correct / len(test_data)}")
        scheduler.step()