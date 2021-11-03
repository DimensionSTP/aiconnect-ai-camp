from facenet_pytorch import MTCNN, InceptionResnetV1, fixed_image_standardization, training
from torchvision import datasets, transforms
import os, torch, copy, cv2, sys, random, logging
from datetime import datetime, timezone, timedelta
from PIL import Image
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
import math

# pip install facenet_pytorch

# Make the Face Datasets
data_dir_train = '/home/dspl-sub/pangyo_ai_camp/pangyo_ai/task01_mask/train'
data_dir_test = '/home/dspl-sub/pangyo_ai_camp/pangyo_ai/task01_mask/test'

batch_size = 32
epochs = 8
workers = 0 if os.name == 'nt' else 8

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('Running on device: {}'.format(device))

mtcnn = MTCNN(
    image_size=160, margin=0, min_face_size=20,
    thresholds=[0.6, 0.7, 0.7], factor=0.709, post_process=True,
    device=device
).eval()

dataset_train = datasets.ImageFolder(data_dir_train, transform=transforms.Resize((512, 512)))
dataset_train.samples = [
    (p, p.replace(data_dir_train, data_dir_train + '_cropped'))
    for p, _ in dataset_train.samples
]
dataset_test = datasets.ImageFolder(data_dir_test, transform=transforms.Resize((512, 512)))
dataset_test.samples = [
    (p, p.replace(data_dir_test, data_dir_test + '_cropped'))
    for p, _ in dataset_test.samples
]

loader_train = DataLoader(
    dataset_train,
    num_workers=workers,
    batch_size=batch_size,
    collate_fn=training.collate_pil
)
loader_test = DataLoader(
    dataset_test,
    num_workers=workers,
    batch_size=batch_size,
    collate_fn=training.collate_pil
)

for i, (x, y) in enumerate(loader_train):
    mtcnn(x, save_path=y)
    print('\rBatch {} of {}'.format(i + 1, len(loader_train)), end='')

for i, (x, y) in enumerate(loader_test):
    mtcnn(x, save_path=y)
    print('\rBatch {} of {}'.format(i + 1, len(loader_test)), end='')

# Remove mtcnn to reduce GPU memory usage
del mtcnn

def get_logger(name: str, file_path: str, stream=False) -> logging.RootLogger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    formatter = logging.Formatter('%(asctime)s | %(name)s | %(levelname)s | %(message)s')
    stream_handler = logging.StreamHandler()
    file_handler = logging.FileHandler(file_path)

    stream_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)

    if stream:
        logger.addHandler(stream_handler)
    logger.addHandler(file_handler)

    return logger

# 시드(seed) 설정

RANDOM_SEED = 2021
torch.manual_seed(RANDOM_SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)

# working directory 지정
ROOT_PATH = '/home/dspl-sub/pangyo_ai_camp'
DATA_DIR = os.path.join(ROOT_PATH, 'pangyo_ai', 'task01_mask', 'train')
DATA_DIR_cropped = os.path.join(ROOT_PATH, 'pangyo_ai', 'task01_mask', 'train_cropped')
RESULT_DIR = os.path.join(ROOT_PATH, 'task01_mask_results')
if not os.path.isdir(RESULT_DIR):
  os.makedirs(RESULT_DIR)

# hyper-parameters
EPOCHS = 25
BATCH_SIZE = 64
LEARNING_RATE = 0.0005
EARLY_STOPPING_PATIENCE = 5
INPUT_SHAPE = (160, 160)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class CustomDataset(Dataset):
    def __init__(self, data_dir, data_dir_cropped, mode, input_shape):

        self.data_dir = data_dir
        self.data_dir_cropped = data_dir_cropped
        self.mode = mode
        self.input_shape = input_shape

        if os.path.isfile(os.path.join(RESULT_DIR, 'total.pkl') and os.path.join(RESULT_DIR, 'total_cropped.pkl')):
            self.db = pd.read_pickle(os.path.join(RESULT_DIR, 'total.pkl'))
            self.db_cropped = pd.read_pickle(os.path.join(RESULT_DIR, 'total_cropped.pkl'))

        else:
            self.db = self.data_loader()
            self.db = self.db.sample(frac=1).reset_index()
            self.db.to_pickle(os.path.join(RESULT_DIR, 'total.pkl'))
            self.db_cropped = self.data_loader_cropped()
            self.db_cropped = self.db_cropped.sample(frac=1).reset_index()
            self.db_cropped.to_pickle(os.path.join(RESULT_DIR, 'total_cropped.pkl'))

        if self.mode == 'train':
            self.db = self.db[:int(len(self.db) * 0.9)]

        elif self.mode == 'val':
            self.db = self.db[int(len(self.db) * 0.9):]
            self.db.reset_index(inplace=True)

        self.transform = transforms.Compose([transforms.Resize(self.input_shape), transforms.ToTensor(),
                                             transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    def data_loader(self):
        print('Loading ' + self.mode + ' dataset..')
        if not os.path.isdir(self.data_dir):
            print(f'!!! Cannot find {self.data_dir}... !!!')
            sys.exit()

        mask_image_list = os.listdir(os.path.join(self.data_dir, 'Mask'))
        nomask_image_list = os.listdir(os.path.join(self.data_dir, 'NoMask'))
        mask_image_list = [item for item in mask_image_list if item[-4:] == '.png']
        nomask_image_list = [item for item in nomask_image_list if item[-4:] == '.png']
        mask_image_path = list(map(lambda x: os.path.join(self.data_dir, 'Mask', x), mask_image_list))
        nomask_image_path = list(map(lambda x: os.path.join(self.data_dir, 'NoMask', x), nomask_image_list))

        # encoding label (Mask : 1, No Mask : 0)
        mask_df = pd.DataFrame({'img_path': mask_image_path, 'label': np.ones(len(mask_image_list))})
        nomask_df = pd.DataFrame({'img_path': nomask_image_path, 'label': np.zeros(len(nomask_image_list))})
        db = mask_df.append(nomask_df, ignore_index=True)
        return db

    def data_loader_cropped(self):
        print('Loading ' + self.mode + ' dataset..')
        if not os.path.isdir(self.data_dir_cropped):
            print(f'!!! Cannot find {self.data_dir_cropped}... !!!')
            sys.exit()

        mask_image_list_cropped = os.listdir(os.path.join(self.data_dir_cropped, 'Mask'))
        nomask_image_list_cropped = os.listdir(os.path.join(self.data_dir_cropped, 'NoMask'))
        mask_image_list_cropped = [item for item in mask_image_list_cropped if item[-4:] == '.png']
        nomask_image_list_cropped = [item for item in nomask_image_list_cropped if item[-4:] == '.png']
        mask_image_path_cropped = list(map(lambda x: os.path.join(self.data_dir_cropped, 'Mask', x), mask_image_list_cropped))
        nomask_image_path_cropped = list(map(lambda x: os.path.join(self.data_dir_cropped, 'NoMask', x), nomask_image_list_cropped))

        # encoding label (Mask : 1, No Mask : 0)
        mask_df_cropped = pd.DataFrame({'img_path': mask_image_path_cropped, 'label': np.ones(len(mask_image_list_cropped))})
        nomask_df_cropped = pd.DataFrame({'img_path': nomask_image_path_cropped, 'label': np.zeros(len(nomask_image_list_cropped))})
        db_cropped = mask_df_cropped.append(nomask_df_cropped, ignore_index=True)
        return db_cropped

    def __len__(self):
        return len(self.db)

    def __getitem__(self, index):
        data = copy.deepcopy(self.db.loc[index])
        # data_cropped = copy.deepcopy(self.db_cropped.loc[index])
        data_cropped = copy.deepcopy(self.db_cropped)
        data_cropped_list = data_cropped['img_path'].str.split('/').str[-1].tolist()

        # Loading image
        cvimg = cv2.imread(data['img_path'], cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)


        # cvimg_cropped = cv2.imread(data_cropped['img_path'], cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)

        if data['img_path'].split('/')[-1] in data_cropped_list:
            idx = data_cropped_list.index(data['img_path'].split('/')[-1])
            cvimg_cropped = cv2.imread(data_cropped['img_path'][idx], cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
        else:
            cvimg_cropped = cvimg

        if not isinstance(cvimg, np.ndarray):
            raise IOError("Fail to read %s" % data['img_path'])

        # Preprocessing images
        trans_image = self.transform(Image.fromarray(cvimg))
        face_image = self.transform(Image.fromarray(cvimg_cropped))
        trans_image = torch.cat((trans_image, face_image), dim=0)
        return trans_image, data['label']


Half_width = 128
layer_width = 128

# define the neural net class
class SpinalVGG(nn.Module):
    def two_conv_pool(self, in_channels, f1, f2):
        s = nn.Sequential(
            nn.Conv2d(in_channels, f1, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(f1),
            nn.ReLU(inplace=True),
            nn.Conv2d(f1, f2, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(f2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        for m in s.children():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
        return s

    def three_conv_pool(self, in_channels, f1, f2, f3):
        s = nn.Sequential(
            nn.Conv2d(in_channels, f1, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(f1),
            nn.ReLU(inplace=True),
            nn.Conv2d(f1, f2, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(f2),
            nn.ReLU(inplace=True),
            nn.Conv2d(f2, f3, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(f3),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        for m in s.children():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
        return s

    def __init__(self, num_classes=2):
        super(SpinalVGG, self).__init__()
        self.l1 = self.two_conv_pool(6, 64, 64)
        self.l2 = self.two_conv_pool(64, 128, 128)
        self.l3 = self.three_conv_pool(128, 256, 256, 256)
        self.l4 = self.three_conv_pool(256, 256, 256, 256)

        self.fc_spinal_layer1 = nn.Sequential(
            nn.Dropout(p=0.5), nn.Linear(Half_width, layer_width),
            nn.BatchNorm1d(layer_width), nn.ReLU(inplace=True), )

        self.fc_spinal_layer2 = nn.Sequential(
            nn.Dropout(p=0.5), nn.Linear(Half_width + layer_width, layer_width),
            nn.BatchNorm1d(layer_width), nn.ReLU(inplace=True), )

        self.fc_spinal_layer3 = nn.Sequential(
            nn.Dropout(p=0.5), nn.Linear(Half_width + layer_width, layer_width),
            nn.BatchNorm1d(layer_width), nn.ReLU(inplace=True), )

        self.fc_spinal_layer4 = nn.Sequential(
            nn.Dropout(p=0.5), nn.Linear(Half_width + layer_width, layer_width),
            nn.BatchNorm1d(layer_width), nn.ReLU(inplace=True), )

        self.fc_out = nn.Sequential(
            nn.Dropout(p=0.5), nn.Linear(layer_width * 4, num_classes), nn.Softmax(dim=1),)

    def forward(self, x):
        x = self.l1(x)
        x = self.l2(x)
        x = self.l3(x)
        x = self.l4(x)
        x = x.view(x.size(0), -1)

        x1 = self.fc_spinal_layer1(x[:, 0:Half_width])
        x2 = self.fc_spinal_layer2(torch.cat([x[:, Half_width:2 * Half_width], x1], dim=1))
        x3 = self.fc_spinal_layer3(torch.cat([x[:, 0:Half_width], x2], dim=1))
        x4 = self.fc_spinal_layer4(torch.cat([x[:, Half_width:2 * Half_width], x3], dim=1))

        x = torch.cat([x1, x2], dim=1)
        x = torch.cat([x, x3], dim=1)
        x = torch.cat([x, x4], dim=1)

        x = self.fc_out(x)

        return x

def conv_start():
    s = nn.Sequential(
        nn.Conv2d(6, 64, kernel_size=7, stride=2, padding=4),
        nn.BatchNorm2d(64),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(kernel_size=3, stride=2),
    )
    for m in s.children():
        if isinstance(m, nn.Conv2d):
            n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            m.weight.data.normal_(0, math.sqrt(2. / n))
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()

    return s

def bottleneck_block(in_dim, mid_dim, out_dim, down=False):
    layers = []
    if down:
        layers.append(nn.Conv2d(in_dim, mid_dim, kernel_size=1, stride=2, padding=0))
    else:
        layers.append(nn.Conv2d(in_dim, mid_dim, kernel_size=1, stride=1, padding=0))
    layers.extend([
        nn.BatchNorm2d(mid_dim),
        nn.ReLU(inplace=True),
        nn.Conv2d(mid_dim, mid_dim, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(mid_dim),
        nn.ReLU(inplace=True),
        nn.Conv2d(mid_dim, out_dim, kernel_size=1, stride=1, padding=0),
        nn.BatchNorm2d(out_dim),
    ])
    return nn.Sequential(*layers)

class Bottleneck(nn.Module):
    def __init__(self, in_dim, mid_dim, out_dim, down:bool = False, starting:bool=False) -> None:
        super(Bottleneck, self).__init__()
        if starting:
            down = False
        self.block = bottleneck_block(in_dim, mid_dim, out_dim, down=down)
        self.relu = nn.ReLU(inplace=True)
        if down:
            self.changedim = nn.Sequential(nn.Conv2d(in_dim, out_dim, kernel_size=1, stride=2, padding=0),
                                           nn.BatchNorm2d(out_dim))
        else:
            self.changedim = nn.Sequential(nn.Conv2d(in_dim, out_dim, kernel_size=1, stride=1, padding=0),
                                           nn.BatchNorm2d(out_dim))

    def forward(self, x):
        identity = self.changedim(x)
        x = self.block(x)
        x += identity
        x = self.relu(x)
        return x

def make_layer(in_dim, mid_dim, out_dim, repeats, starting=False):
    layers = []
    layers.append(Bottleneck(in_dim, mid_dim, out_dim, down=True, starting=starting))
    for _ in range(1, repeats):
        layers.append(Bottleneck(out_dim, mid_dim, out_dim, down=False))
    return nn.Sequential(*layers)


class SpinalResNet(nn.Module):
    def __init__(self, num_classes=2):
        super(SpinalResNet, self).__init__()
        self.num_classes = num_classes
        # 1번
        self.conv1 = conv_start()
        repeats = [3, 4, 6, 3]
        # 2번
        base_dim = 64

        self.conv2 = make_layer(base_dim, base_dim, base_dim * 4, repeats[0], starting=True)
        self.conv3 = make_layer(base_dim * 4, base_dim * 2, base_dim * 8, repeats[1])
        self.conv4 = make_layer(base_dim * 8, base_dim * 4, base_dim * 16, repeats[2])
        self.conv5 = make_layer(base_dim * 16, base_dim * 8, base_dim * 32, repeats[3])

        # 3번
        # self.avgpool = nn.AvgPool2d(kernel_size=5, stride=1)
        self.fc_spinal_layer1 = nn.Sequential(
            nn.Dropout(p=0.5), nn.Linear(Half_width, layer_width),
            nn.BatchNorm1d(layer_width), nn.ReLU(inplace=True), )

        self.fc_spinal_layer2 = nn.Sequential(
            nn.Dropout(p=0.5), nn.Linear(Half_width + layer_width, layer_width),
            nn.BatchNorm1d(layer_width), nn.ReLU(inplace=True), )

        self.fc_spinal_layer3 = nn.Sequential(
            nn.Dropout(p=0.5), nn.Linear(Half_width + layer_width, layer_width),
            nn.BatchNorm1d(layer_width), nn.ReLU(inplace=True), )

        self.fc_spinal_layer4 = nn.Sequential(
            nn.Dropout(p=0.5), nn.Linear(Half_width + layer_width, layer_width),
            nn.BatchNorm1d(layer_width), nn.ReLU(inplace=True), )

        self.fc_out = nn.Sequential(
            nn.Dropout(p=0.5), nn.Linear(layer_width * 4, num_classes), nn.Softmax(dim=1), )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        # x = self.avgpool(x)

        x = x.view(x.size(0), -1)

        x1 = self.fc_spinal_layer1(x[:, 0:Half_width])
        x2 = self.fc_spinal_layer2(torch.cat([x[:, Half_width:2 * Half_width], x1], dim=1))
        x3 = self.fc_spinal_layer3(torch.cat([x[:, 0:Half_width], x2], dim=1))
        x4 = self.fc_spinal_layer4(torch.cat([x[:, Half_width:2 * Half_width], x3], dim=1))

        x = torch.cat([x1, x2], dim=1)
        x = torch.cat([x, x3], dim=1)
        x = torch.cat([x, x4], dim=1)

        x = self.fc_out(x)

        return x

class LossEarlyStopper():
    """Early stopper

    Attributes:
        patience (int): loss가 줄어들지 않아도 학습할 epoch 수
        verbose (bool): 로그 출력 여부, True 일 때 로그 출력
        patience_counter (int): loss 가 줄어들지 않을 때 마다 1씩 증가
        min_loss (float): 최소 loss
        stop (bool): True 일 때 학습 중단

    """

    def __init__(self, patience: int, verbose: bool, logger: logging.RootLogger = None) -> None:
        """ 초기화

        Args:
            patience (int): loss가 줄어들지 않아도 학습할 epoch 수
            weight_path (str): weight 저장경로
            verbose (bool): 로그 출력 여부, True 일 때 로그 출력
        """
        self.patience = patience
        self.verbose = verbose

        self.patience_counter = 0
        self.min_loss = np.Inf
        self.logger = logger
        self.stop = False

    def check_early_stopping(self, loss: float) -> None:
        """Early stopping 여부 판단

        Args:
            loss (float):

        Examples:

        Note:

        """

        if self.min_loss == np.Inf:
            self.min_loss = loss
            # self.save_checkpoint(loss=loss, model=model)

        elif loss > self.min_loss:
            self.patience_counter += 1
            msg = f"Early stopper, Early stopping counter {self.patience_counter}/{self.patience}"

            if self.patience_counter == self.patience:
                self.stop = True

            if self.verbose:
                self.logger.info(msg) if self.logger else print(msg)

        elif loss <= self.min_loss:
            self.save_model = True
            msg = f"Early stopper, Validation loss decreased {self.min_loss} -> {loss}"
            self.min_loss = loss
            # self.save_checkpoint(loss=loss, model=model)

            if self.verbose:
                self.logger.info(msg) if self.logger else print(msg)


class Trainer():
    """ Trainer
        epoch에 대한 학습 및 검증 절차 정의
    """
    def __init__(self, criterion, model, device, metric_fn, optimizer=None, logger=None):
        """ 초기화
        """
        self.criterion = criterion
        self.model = model
        self.device = device
        self.optimizer = optimizer
        self.logger = logger
        self.metric_fn = metric_fn

    def train_epoch(self, dataloader, epoch_index):
        """ 한 epoch에서 수행되는 학습 절차
        """
        self.model.train()
        train_total_loss = 0
        target_lst = []
        pred_lst = []
        prob_lst = []

        for batch_index, (img, label) in enumerate(dataloader):
            img = img.to(self.device)
            label = label.to(self.device).long()
            pred = self.model(img)
            loss = self.criterion(pred, label)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            train_total_loss += loss.item()
            prob_lst.extend(pred[:, 1].cpu().tolist())
            target_lst.extend(label.cpu().tolist())
            pred_lst.extend(pred.argmax(dim=1).cpu().tolist())
        self.train_mean_loss = train_total_loss / batch_index
        self.train_score, auroc = self.metric_fn(y_pred=pred_lst, y_answer=target_lst, y_prob=prob_lst)
        msg = f'Epoch {epoch_index}, Train loss: {self.train_mean_loss}, Acc: {self.train_score}, ROC: {auroc}'
        print(msg)

    def validate_epoch(self, dataloader, epoch_index):
        """ 한 epoch에서 수행되는 검증 절차
        """
        self.model.eval()
        val_total_loss = 0
        target_lst = []
        pred_lst = []
        prob_lst = []

        for batch_index, (img, label) in enumerate(dataloader):
            img = img.to(self.device)
            label = label.to(self.device).long()
            pred = self.model(img)
            ## coordinate loss
            loss = self.criterion(pred, label)
            val_total_loss += loss.item()
            prob_lst.extend(pred[:, 1].cpu().tolist())
            target_lst.extend(label.cpu().tolist())
            pred_lst.extend(pred.argmax(dim=1).cpu().tolist())
        self.val_mean_loss = val_total_loss / batch_index
        self.validation_score, auroc = self.metric_fn(y_pred=pred_lst, y_answer=target_lst, y_prob=prob_lst)
        msg = f'Epoch {epoch_index}, Val loss: {self.val_mean_loss}, Acc: {self.validation_score}, ROC: {auroc}'
        print(msg)

from sklearn.metrics import accuracy_score, roc_auc_score

def get_metric_fn(y_pred, y_answer, y_prob):
    """ 성능을 반환하는 함수
    """
    assert len(y_pred) == len(y_answer), 'The size of prediction and answer are not same.'
    accuracy = accuracy_score(y_answer, y_pred)
    auroc = roc_auc_score(y_answer, y_prob)
    return accuracy, auroc

# Load dataset & dataloader
train_dataset = CustomDataset(data_dir=DATA_DIR, data_dir_cropped=DATA_DIR_cropped, mode='train', input_shape=INPUT_SHAPE)
validation_dataset = CustomDataset(data_dir=DATA_DIR, data_dir_cropped=DATA_DIR_cropped, mode='val', input_shape=INPUT_SHAPE)

train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
validation_dataloader = DataLoader(validation_dataset, batch_size=BATCH_SIZE, shuffle=True)
print('Train set samples:',len(train_dataset),  'Val set samples:', len(validation_dataset))

# Load Model
model = SpinalResNet().to(device)
model2 = SpinalVGG().to(device)

# Save Initial Model
torch.save({'model':model.state_dict()}, os.path.join(RESULT_DIR, 'initial.pt'))
torch.save({'model':model2.state_dict()}, os.path.join(RESULT_DIR, 'initial_2.pt'))
# Set optimizer, scheduler, loss function, metric function
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
optimizer2 = optim.Adam(model2.parameters(), lr=LEARNING_RATE)
criterion = nn.CrossEntropyLoss()
criterion2 = nn.CrossEntropyLoss()

metric_fn = get_metric_fn

# Set system logger
system_logger = get_logger(name='train',file_path='train_log.log')

# Set trainer
trainer = Trainer(criterion, model, device, metric_fn, optimizer, logger=system_logger)
trainer2 = Trainer(criterion2, model2, device, metric_fn, optimizer2, logger=system_logger)
early_stopper = LossEarlyStopper(patience=EARLY_STOPPING_PATIENCE, verbose=True, logger=system_logger)

criterion = 1E+8
for epoch_index in tqdm(range(EPOCHS)):

    trainer.train_epoch(train_dataloader, epoch_index)
    trainer.validate_epoch(validation_dataloader, epoch_index)

    if trainer.val_mean_loss < criterion:
        criterion = trainer.val_mean_loss
        check_point = {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict()
        }
        torch.save(check_point, os.path.join(RESULT_DIR, 'best.pt'))

TRAINED_MODEL_PATH = os.path.join(RESULT_DIR, 'best.pt')

criterion = 1E+8
for epoch_index in tqdm(range(EPOCHS)):

    trainer2.train_epoch(train_dataloader, epoch_index)
    trainer2.validate_epoch(validation_dataloader, epoch_index)

    if trainer2.val_mean_loss < criterion:
        criterion = trainer2.val_mean_loss
        check_point = {
            'model': model2.state_dict(),
            'optimizer': optimizer2.state_dict()
        }
        torch.save(check_point, os.path.join(RESULT_DIR, 'best_2.pt'))

TRAINED_MODEL_PATH2 = os.path.join(RESULT_DIR, 'best_2.pt')

class TestDataset(Dataset):
    def __init__(self, data_dir, data_dir_cropped, input_shape):
        self.data_dir = data_dir
        self.data_dir_cropped = data_dir_cropped
        self.input_shape = input_shape

        # Loading dataset
        self.db = self.data_loader()
        self.db_cropped = self.data_loader_cropped()
        # Transform function
        self.transform = transforms.Compose([transforms.Resize(self.input_shape), transforms.ToTensor(),
                                             transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    def data_loader(self):
        print('Loading test dataset..')
        if not os.path.isdir(self.data_dir):
            print(f'!!! Cannot find {self.data_dir}... !!!')
            sys.exit()
        image_list = os.listdir(self.data_dir)
        image_list = [item for item in image_list if item[-4:] == '.png']
        image_path = list(map(lambda x: os.path.join(self.data_dir, x), image_list))
        db = pd.DataFrame({'img_path': image_path, 'file_num': list(map(lambda x: x.split('.')[0], image_list))})
        return db

    def data_loader_cropped(self):
        print('Loading test dataset..')
        if not os.path.isdir(self.data_dir_cropped):
            print(f'!!! Cannot find {self.data_dir_cropped}... !!!')
            sys.exit()
        image_list_cropped = os.listdir(self.data_dir_cropped)
        image_list_cropped = [item for item in image_list_cropped if item[-4:] == '.png']
        image_path_cropped = list(map(lambda x: os.path.join(self.data_dir_cropped, x), image_list_cropped))
        db_cropped = pd.DataFrame({'img_path': image_path_cropped, 'file_num': list(map(lambda x: x.split('.')[0], image_list_cropped))})
        return db_cropped

    def __len__(self):
        return len(self.db)

    def __getitem__(self, index):
        data = copy.deepcopy(self.db.loc[index])
        data_cropped = copy.deepcopy(self.db_cropped)

        # Loading image
        cvimg = cv2.imread(data['img_path'], cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
        # cvimg_cropped = cv2.imread(data_cropped['img_path'], cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)


        if not isinstance(cvimg, np.ndarray):
            raise IOError("Fail to read %s" % data['img_path'])

        data_cropped_list = data_cropped['img_path'].str.split('/').str[-1].tolist()

        if data['img_path'].split('/')[-1] in data_cropped_list:
            idx = data_cropped_list.index(data['img_path'].split('/')[-1])
            cvimg_cropped = cv2.imread(data_cropped['img_path'][idx], cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
        else:
            cvimg_cropped = cvimg

        # Preprocessing images
        trans_image = self.transform(Image.fromarray(cvimg))
        face_image = self.transform(Image.fromarray(cvimg_cropped))
        trans_image = torch.cat((trans_image, face_image), dim=0)

        return trans_image, data['img_path'], data['file_num']

DATA_DIR=os.path.join(ROOT_PATH, 'pangyo_ai', 'task01_mask', 'test', '05_final_test')
DATA_DIR_cropped=os.path.join(ROOT_PATH, 'pangyo_ai', 'task01_mask', 'test_cropped', '05_final_test')
# Load dataset & dataloader
test_dataset = TestDataset(data_dir=DATA_DIR, data_dir_cropped=DATA_DIR_cropped, input_shape=INPUT_SHAPE)
test_dataloader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=False)

model.load_state_dict(torch.load(TRAINED_MODEL_PATH)['model'])
model2.load_state_dict(torch.load(TRAINED_MODEL_PATH2)['model'])

# Prediction
file_num_lst = []
pred_lst = []
prob_lst = []
path_lst = []
model.eval()
model2.eval()

with torch.no_grad():
    for batch_index, (img, path, file_num) in enumerate(test_dataloader):
        img = img.to(device)
        pred = model(img)
        pred2 = model2(img)
        pred_total = pred + pred2
        file_num_lst.extend(list(file_num))
        path_lst.extend(path)
        pred_lst.extend(pred_total.argmax(dim=1).tolist())
        prob_lst.extend(pred_total[:, 1].tolist())

df = pd.DataFrame({'file_name':list(map(int, file_num_lst)), 'path':path_lst, 'answer':pred_lst, 'prob':prob_lst})
df.sort_values(by=['file_name'], inplace=True)
# df.to_csv(os.path.join(RESULT_DIR, 'mask_pred.csv'), index=False)

class pseudo_CustomDataset(Dataset):
    def __init__(self, data_dir, data_dir_cropped, testdataset, data_dir_pseudo_cropped, mode, input_shape):

        self.data_dir = data_dir
        self.data_dir_cropped = data_dir_cropped
        self.mode = mode
        self.input_shape = input_shape
        self.testdataset = testdataset
        self.data_dir_pseudo_cropped = data_dir_pseudo_cropped

        if os.path.isfile(os.path.join(RESULT_DIR, 'total_pseudo.pkl')
                          and os.path.join(RESULT_DIR, 'total_cropped_pseudo.pkl')):
            self.db = pd.read_pickle(os.path.join(RESULT_DIR, 'total_pseudo.pkl'))
            self.db_cropped = pd.read_pickle(os.path.join(RESULT_DIR, 'total_cropped_pseudo.pkl'))

        else:
            self.db = self.data_loader()
            self.db = self.db.sample(frac=1).reset_index()
            self.db.to_pickle(os.path.join(RESULT_DIR, 'total_pseudo.pkl'))
            self.db_cropped = self.data_loader_cropped()
            self.db_cropped = self.db_cropped.sample(frac=1).reset_index()
            self.db_cropped.to_pickle(os.path.join(RESULT_DIR, 'total_cropped_pseudo.pkl'))

        if self.mode == 'train':
            self.db = pd.concat([self.db[:int(27098 * 0.9)], self.db[-1026:]])
            self.db.reset_index(inplace=True)

        elif self.mode == 'val':
            self.db = self.db[int(27098 * 0.9): -1026]
            self.db.reset_index(inplace=True)

        self.transform = transforms.Compose([transforms.Resize(self.input_shape), transforms.ToTensor(),
                                             transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    def data_loader(self):
        print('Loading ' + self.mode + ' dataset..')
        if not os.path.isdir(self.data_dir):
            print(f'!!! Cannot find {self.data_dir}... !!!')
            sys.exit()

        mask_image_list = os.listdir(os.path.join(self.data_dir, 'Mask'))
        nomask_image_list = os.listdir(os.path.join(self.data_dir, 'NoMask'))
        mask_image_list = [item for item in mask_image_list if item[-4:] == '.png']
        nomask_image_list = [item for item in nomask_image_list if item[-4:] == '.png']
        mask_image_path = list(map(lambda x: os.path.join(self.data_dir, 'Mask', x), mask_image_list))
        nomask_image_path = list(map(lambda x: os.path.join(self.data_dir, 'NoMask', x), nomask_image_list))

        # encoding label (Mask : 1, No Mask : 0)
        mask_df = pd.DataFrame({'img_path': mask_image_path, 'label': np.ones(len(mask_image_list))})
        nomask_df = pd.DataFrame({'img_path': nomask_image_path, 'label': np.zeros(len(nomask_image_list))})
        db = mask_df.append(nomask_df, ignore_index=True)

        test_df = pd.DataFrame({'img_path': list(self.testdataset['path']), 'label': list(self.testdataset['answer'])})

        db = db.append(test_df, ignore_index=True)
        return db

    def data_loader_cropped(self):
        print('Loading ' + self.mode + ' dataset..')
        if not os.path.isdir(self.data_dir_cropped):
            print(f'!!! Cannot find {self.data_dir_cropped}... !!!')
            sys.exit()

        mask_image_list_cropped = os.listdir(os.path.join(self.data_dir_cropped, 'Mask'))
        nomask_image_list_cropped = os.listdir(os.path.join(self.data_dir_cropped, 'NoMask'))
        mask_image_list_cropped = [item for item in mask_image_list_cropped if item[-4:] == '.png']
        nomask_image_list_cropped = [item for item in nomask_image_list_cropped if item[-4:] == '.png']
        mask_image_path_cropped = list(map(lambda x: os.path.join(self.data_dir_cropped, 'Mask', x), mask_image_list_cropped))
        nomask_image_path_cropped = list(map(lambda x: os.path.join(self.data_dir_cropped, 'NoMask', x), nomask_image_list_cropped))

        # encoding label (Mask : 1, No Mask : 0)
        mask_df_cropped = pd.DataFrame({'img_path': mask_image_path_cropped, 'label': np.ones(len(mask_image_list_cropped))})
        nomask_df_cropped = pd.DataFrame({'img_path': nomask_image_path_cropped, 'label': np.zeros(len(nomask_image_list_cropped))})
        db_cropped = mask_df_cropped.append(nomask_df_cropped, ignore_index=True)

        image_list_pseudo_cropped = os.listdir(os.path.join(self.data_dir_pseudo_cropped))
        image_list_pseudo_cropped = [item for item in image_list_pseudo_cropped if item[-4:] == '.png']
        image_path_pseudo_cropped = list(
            map(lambda x: os.path.join(self.data_dir_pseudo_cropped, x), image_list_pseudo_cropped))

        db_pseudo_cropped = pd.DataFrame({'img_path': image_path_pseudo_cropped})
        cropped_name_lst = list(db_pseudo_cropped['img_path'].str.split('/').str[-1])
        test_name_lst = list(self.testdataset['path'].str.split('/').str[-1])

        img_path = []
        label = []
        for i in cropped_name_lst:
            if i in test_name_lst:
                pos_idx = test_name_lst.index(i)
                img_path.append(list(self.testdataset['path'])[pos_idx])
                label.append(list(self.testdataset['answer'])[pos_idx])
                # pseudo_df_cropped = pd.DataFrame({'img_path': list(self.testdataset['path'])[pos_idx], 'label': list(self.testdataset['answer'])[pos_idx]})
                # db_cropped.append(pseudo_df_cropped, ignore_index=True)

        pseudo_df_cropped = pd.DataFrame({'img_path': img_path, 'label': label})
        db_cropped.append(pseudo_df_cropped, ignore_index=True)
        return db_cropped

    def __len__(self):
        return len(self.db)

    def __getitem__(self, index):
        data = copy.deepcopy(self.db.loc[index])
        # data_cropped = copy.deepcopy(self.db_cropped.loc[index])
        data_cropped = copy.deepcopy(self.db_cropped)
        data_cropped_list = data_cropped['img_path'].str.split('/').str[-1].tolist()

        # Loading image
        cvimg = cv2.imread(data['img_path'], cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)

        if data['img_path'].split('/')[-1] in data_cropped_list:
            idx = data_cropped_list.index(data['img_path'].split('/')[-1])
            cvimg_cropped = cv2.imread(data_cropped['img_path'][idx], cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
        else:
            cvimg_cropped = cvimg

        if not isinstance(cvimg, np.ndarray):
            raise IOError("Fail to read %s" % data['img_path'])

        # Preprocessing images
        trans_image = self.transform(Image.fromarray(cvimg))
        face_image = self.transform(Image.fromarray(cvimg_cropped))
        trans_image = torch.cat((trans_image, face_image), dim=0)
        return trans_image, data['label']


DATA_DIR = os.path.join(ROOT_PATH, 'pangyo_ai', 'task01_mask', 'train')
DATA_DIR_cropped = os.path.join(ROOT_PATH, 'pangyo_ai', 'task01_mask', 'train_cropped')
RESULT_DIR = os.path.join(ROOT_PATH, 'task01_mask_results')
DATA_DIR_cropped_pseudo = os.path.join(ROOT_PATH, 'pangyo_ai', 'task01_mask', 'test_cropped', '05_final_test')

train_dataset_pseudo = pseudo_CustomDataset(data_dir=DATA_DIR, data_dir_cropped=DATA_DIR_cropped, testdataset=df,
                                            data_dir_pseudo_cropped=DATA_DIR_cropped_pseudo, mode='train',
                                            input_shape=INPUT_SHAPE)
validation_dataset_pseudo = pseudo_CustomDataset(data_dir=DATA_DIR, data_dir_cropped=DATA_DIR_cropped, testdataset=df,
                                            data_dir_pseudo_cropped=DATA_DIR_cropped_pseudo, mode='val',
                                            input_shape=INPUT_SHAPE)
train_dataloader = DataLoader(train_dataset_pseudo, batch_size=BATCH_SIZE, shuffle=True)
validation_dataloader = DataLoader(validation_dataset_pseudo, batch_size=BATCH_SIZE, shuffle=True)

print('Pseudo Train set samples:',len(train_dataset_pseudo),  'Pseudo Val set samples:', len(validation_dataset_pseudo))

model.load_state_dict(torch.load(TRAINED_MODEL_PATH)['model'])
model2.load_state_dict(torch.load(TRAINED_MODEL_PATH2)['model'])

# Set optimizer, scheduler, loss function, metric function
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
optimizer2 = optim.Adam(model2.parameters(), lr=LEARNING_RATE)
criterion = nn.CrossEntropyLoss()
criterion2 = nn.CrossEntropyLoss()
metric_fn = get_metric_fn

# Set system logger
system_logger = get_logger(name='train', file_path='train_log.log')

trainer = Trainer(criterion, model, device, metric_fn, optimizer, logger=system_logger)
trainer2 = Trainer(criterion2, model2, device, metric_fn, optimizer2, logger=system_logger)

criterion = 1E+8
for epoch_index in tqdm(range(EPOCHS)):
    trainer.train_epoch(train_dataloader, epoch_index)
    trainer.validate_epoch(validation_dataloader, epoch_index)

    if trainer.val_mean_loss < criterion:
        criterion = trainer.val_mean_loss
        check_point = {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict()
        }
        torch.save(check_point, os.path.join(RESULT_DIR, 'best_pseudo.pt'))

TRAINED_MODEL_PATH_pseudo = os.path.join(RESULT_DIR, 'best_pseudo.pt')

criterion = 1E+8
for epoch_index in tqdm(range(EPOCHS)):
    trainer2.train_epoch(train_dataloader, epoch_index)
    trainer2.validate_epoch(validation_dataloader, epoch_index)

    if trainer2.val_mean_loss < criterion:
        criterion = trainer2.val_mean_loss
        check_point = {
            'model': model2.state_dict(),
            'optimizer': optimizer2.state_dict()
        }
        torch.save(check_point, os.path.join(RESULT_DIR, 'best_pseudo_2.pt'))

TRAINED_MODEL_PATH_pseudo2 = os.path.join(RESULT_DIR, 'best_pseudo_2.pt')

DATA_DIR = os.path.join(ROOT_PATH, 'pangyo_ai', 'task01_mask', 'test', '05_final_test')
DATA_DIR_cropped = os.path.join(ROOT_PATH, 'pangyo_ai', 'task01_mask', 'test_cropped', '05_final_test')

# Load dataset & dataloader
test_dataset = TestDataset(data_dir=DATA_DIR, data_dir_cropped=DATA_DIR_cropped, input_shape=INPUT_SHAPE)
test_dataloader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=False)

model.load_state_dict(torch.load(TRAINED_MODEL_PATH_pseudo)['model'])
model2.load_state_dict(torch.load(TRAINED_MODEL_PATH_pseudo2)['model'])

# Prediction
file_num_lst = []
pred_lst_total = []
prob_lst_total = []
model.eval()
model2.eval()
with torch.no_grad():
    for batch_index, (img, _, file_num) in enumerate(test_dataloader):
        img = img.to(device)
        pred = model(img)
        pred2 = model2(img)
        pred_total = pred + pred2 / 2
        file_num_lst.extend(list(file_num))
        pred_lst_total.extend(pred_total.argmax(dim=1).tolist())
        prob_lst_total.extend(pred_total[:, 1].tolist())


df_total = pd.DataFrame({'file_name': list(map(int, file_num_lst)), 'answer': pred_lst_total, 'prob': prob_lst_total})
df_total.sort_values(by=['file_name'], inplace=True)
df_total.to_csv(os.path.join(RESULT_DIR, 'mask_pred.csv'), index=False)