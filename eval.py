import torch
from torch.utils import data
from data import TestDataset
from model import Model
import os

save_path = '/media/disk1/pxq/ASD/'

test_dataset = TestDataset()
test_loader = data.DataLoader(test_dataset, batch_size=1, shuffle=False)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = Model().to(device)
checkpoint = torch.load(os.path.join(save_path, 'checkpoint_best.pth'))
model.load_state_dict(checkpoint['model'])
print('Epoch:{} MAE:{} MSE:{}'.format(checkpoint['epoch'], checkpoint['mae'], checkpoint['mse']))

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
    print('MAE:{} MSE:{}'.format(mae, mse))
