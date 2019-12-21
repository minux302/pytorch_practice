from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torchvision
from torchvision import models, transforms

from dataset import ImageTransform, make_datapath_list, HymenopteraDataset


def train_model(num_epochs):
    device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')
    print("device: ", device)
    torch.backends.cudnn.benchmark = True

    # dataset preparation
    batch_size = 32
    train_list = make_datapath_list(phase="train")
    val_list = make_datapath_list(phase="val")

    size = 224
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)

    train_dataset = HymenopteraDataset(
        file_list=train_list,
        transform=ImageTransform(size, mean, std),
        phase='train')
    val_dataset = HymenopteraDataset(
        file_list=val_list,
        transform=ImageTransform(size, mean, std),
        phase='val')
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False)
    dataloaders_dict = {'train': train_dataloader, 'val': val_dataloader}

    net = models.vgg16(pretrained=True)
    net.classifier[6] = nn.Linear(in_features=4096, out_features=2)
    net.to(device)
    criterion = nn.CrossEntropyLoss()
    params_to_update = []
    update_param_names = ['classifier.6.weight', 'classifier.6.bias']
    for name, param in net.named_parameters():
        if name in update_param_names:
            param.required_grad = True
            params_to_update.append(param)
            print(name)
        else:
            param.required_grad = False
    optimizer = optim.SGD(params=params_to_update, lr=0.001, momentum=0.9)

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch+1, num_epochs))
        print('--------------------------')

        for phase in ['train', 'val']:
            if phase == 'train':
                net.train()
            else:
                net.eval()

            epoch_loss = 0.0
            epoch_corrects = 0

            if (epoch == 0) and (phase == 'train'):
                continue

            # for inputs, labels in tqdm(dataloaders_dict[phase]):
            for inputs, labels in tqdm(train_dataloader):
                inputs = inputs.to(device)
                labels = labels.to(device)
                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = net(inputs)
                    loss = criterion(outputs, labels)
                    _, preds = torch.max(outputs, 1)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                    epoch_loss += loss.item() * inputs.size(0)
                    epoch_corrects += torch.sum(preds == labels.data)

        epoch_loss = epoch_loss / len(dataloaders_dict[phase].dataset)
        epoch_acc = epoch_corrects.double() / len(dataloaders_dict[phase].dataset)
        print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))


if __name__ == '__main__':
    num_epochs = 2
    train_model(num_epochs)