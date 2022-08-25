import torch
from train import per_epoch_activity
from utils import load_data_bananas, get_device, multibox_target, calc_loss, Accumulator,cls_eval, bbox_eval
from torch.utils.tensorboard import SummaryWriter
from module import SSD
import time

EPOCH = 16
batch_size = 32
train_iter, val_iter = load_data_bananas(batch_size)
device = get_device()


loss_calculator= calc_loss
model = SSD(num_classes=1)
model.to(device)

optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
accumulator = Accumulator
summary_writer = SummaryWriter()
if __name__ == '__main__':
    per_epoch_activity(EPOCH, train_iter, model, multibox_target, loss_calculator, optimizer,
                       accumulator, device, summary_writer, cls_eval, bbox_eval)
