import torch
from torch import nn
from torch import optim
from torch.utils import data
from data import TrainDataset, TestDataset
from model import Model
from logger import Logger
import os

batch_size = 16
end_epoch = 1000
load_checkpoint = False
save_path = '/media/disk1/pxq/ASD/'

train_dataset = TrainDataset()
train_loader = data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_dataset = TestDataset()
test_loader = data.DataLoader(test_dataset, batch_size=1, shuffle=False)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = Model().to(device)

criterion = nn.MSELoss(size_average=False).to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-4)

loss_logger = Logger('./logs/loss')
eval_logger = Logger('./logs/eval')

if load_checkpoint:
    checkpoint = torch.load(os.path.join(save_path, 'checkpoint_latest.pth'))
    model.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optimizer'])

    best_mae = torch.load(os.path.join(save_path, 'checkpoint_best.pth'))['mae']
    start_epoch = checkpoint['epoch'] + 1
else:
    best_mae = 999999
    start_epoch = 0

for epoch in range(start_epoch, end_epoch):
    loss_avg = 0.0
    for i, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        print('output:{:.2f} label:{:.2f}'.format(outputs.sum().item() / batch_size, labels.sum().item() / batch_size))

        loss = criterion(outputs, labels) / batch_size / 2

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_avg += loss.item()

        print("Epoch:{}, Step:{}, Loss:{:.4f}({:.4f})".format(epoch, i, loss.item(), loss_avg / (i + 1)))

    loss_logger.scalar_summary('loss_avg', loss_avg / len(train_loader), epoch)

    model.eval()
    with torch.no_grad():
        mae, mse, count = 0.0, 0.0, 0
        for i, (images, labels) in enumerate(test_loader):
            images = images.to(device)
            labels = labels.to(device)

            predict = model(images)

            print('predict:{:.2f} label:{:.2f}'.format(predict.sum().item(), labels.sum().item()))
            mae += torch.abs(predict.sum() - labels.sum()).item()
            mse += ((predict.sum() - labels.sum()) ** 2).item()
            count += 1

        mae /= count
        mse /= count
        mse = mse ** 0.5
        print('Epoch:', epoch, 'MAE:', mae, 'MSE:', mse)
        eval_logger.scalar_summary('MAE', mae, epoch)
        eval_logger.scalar_summary('MSE', mse, epoch)

        state = {'epoch': epoch, 'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'mae': mae,
                 'mse': mse}
        torch.save(state, os.path.join(save_path, 'checkpoint_latest.pth'))

        if mae < best_mae:
            best_mae = mae
            torch.save(state, os.path.join(save_path, 'checkpoint_best.pth'))
    model.train()
