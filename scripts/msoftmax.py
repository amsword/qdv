#import matplotlib
#matplotlib.use('Agg')
from matplotlib import pyplot as plt
import torch
import torch.functional as F
import torchvision
import torchvision.transforms as transforms
import numpy as np
import logging
import random
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from qd_common import ensure_directory
import os
import os.path as op
from scipy import linalg as LA
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
from pprint import pformat

def init_logging():
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    torch.random.manual_seed(777)
    torch.manual_seed(777)
    torch.cuda.manual_seed(777)
    import random
    random.seed(777)
    np.seterr(all='raise')
    np.random.seed(777)
    logging.basicConfig(level=logging.INFO,
        format='%(asctime)s.%(msecs)03d %(filename)s:%(lineno)s %(funcName)10s(): %(message)s',
        datefmt='%m-%d %H:%M:%S',
    )

def get_data(num_real_classes, num_sample, radius, sigma, class_grad):
    degree_per_class = 360. / num_real_classes
    sigma_maxtrix = np.zeros((2, 2))
    sigma_maxtrix[0, 0] = sigma * sigma
    sigma_maxtrix[-1, -1] = sigma * sigma
    num_train_sample_per_real_class = num_sample / num_real_classes
    def create_feat_label(num_sample_per_real_class):
        features = []
        labels = []
        for c, g in enumerate(class_grad):
            curr_degree = g * degree_per_class
            mu_x = radius * np.cos(curr_degree / 180. * np.pi)
            mu_y = radius * np.sin(curr_degree / 180. * np.pi)
        
            curr_samples = np.random.multivariate_normal([mu_x, mu_y],
                    sigma_maxtrix, num_sample_per_real_class)
            curr_labels = np.zeros((num_sample_per_real_class), dtype=np.int64)
            curr_labels[:] = c
            features.append(curr_samples)
            labels.append(curr_labels)
        feat, label = np.vstack(features), np.hstack(labels)
        n = len(feat)
        idx = range(n)
        random.shuffle(idx)
        return feat[idx, :], label[idx]
    train_features, train_labels = create_feat_label(num_train_sample_per_real_class)
    return train_features, train_labels

class Net(nn.Module):
    def __init__(self, num_feature, num_output):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(num_feature, num_output, bias=False)

    def forward(self, x):
        return self.fc1(x)


def gen_colors(num_real_classes):
    colors = []
    for c in range(num_real_classes):
        colors.append(np.random.rand(3))
    return colors

def plot(net, train_features, train_labels, colors, radius, ax):
    num_inferred_classes = len(set(train_labels[:]))
    # training samples
    for c in range(num_inferred_classes):
        is_curr = train_labels == c
        ax.scatter(train_features[is_curr, 0], 
                train_features[is_curr, 1],
                c=colors[c])
    
    all_param = list(net.parameters())
    assert len(all_param) == 1
    w = all_param[0].data.cpu().numpy()
    
    logging.info(w) 
    for i, (x, y) in enumerate(w):
        ratio = radius / np.sqrt(x * x + y * y)
        ax.plot([0, ratio * x], [0, ratio * y], color=colors[i])
        ax.scatter([x], [y], color=colors[i])
    ax.grid()

class NMLoss(nn.Module):
    def __init__(self, N, M, loss_type, num_proto=1):
        super(NMLoss, self).__init__()
        self.N = N
        self.M = M
        self.eps = 0.00001
        self.loss_type = loss_type
        self._iter = 0
        self._ce = nn.CrossEntropyLoss()
        self._num_proto = num_proto
        self._multi_policy_proto = 'max_softmax' # softmax_sum, softmax_max

    def forward(self, pred, target):
        '''
        pred should be the linear output. softmax will be calculated here
        '''
        batch_size = pred.data.size(0)
        pred= pred.view(batch_size, self.M + self._num_proto * self.N)
        if self._num_proto > 1:
            if self._multi_policy_proto == 'max_softmax':
                first = pred[:, :(self.N * self._num_proto)].contiguous().view(batch_size, self.N,
                        self._num_proto)
                first_max, _ = torch.max(first, dim=2)
                second = pred[:, (self.N * self._num_proto): ]
                pred = torch.cat((first_max, second), dim=1)
                prediction = F.softmax(pred, dim=1)
            elif self._multi_policy_proto == 'softmax_sum':
                prediction = F.softmax(pred, dim=1)
                first = prediction[:, :(self.N *
                    self._num_proto)].contiguous().view(batch_size, self.N,
                        self._num_proto)
                first_sum = torch.sum(first, dim=2)
                second = prediction[:, (self.N * self._num_proto): ]
                prediction = torch.cat((first_max, second), dim=1)
        else:
            prediction = F.softmax(pred, dim=1)
        
        loss = 0

        # cross entropy loss
        loss_ce = 0
        if 'cross_entropy' in self.loss_type:
            prob_N = prediction.index_select(1, 
                    torch.autograd.Variable(torch.arange(0, self.N).long().cuda()))
            prob_M = prediction.index_select(1,
                    torch.autograd.Variable(torch.arange(self.N, self.N +
                        self.M).long().cuda()))
            prob_sM = torch.sum(prob_M, 1, keepdim=True)
            prob_N1 = torch.cat((prob_N, prob_sM), dim=1)
            log_prob_N1 = torch.log(prob_N1 + self.eps)
            loss_ce = F.nll_loss(log_prob_N1, target)
            loss += loss_ce * self.loss_type.get('cross_entropy', 1)

        # entropy loss
        loss_en = 0
        if 'entropy_loss' in self.loss_type or \
                'uniform_loss' in self.loss_type:
            negative_prob_M = prob_M[(target.data == self.N).nonzero().squeeze(1), :]
            norm_neg_prob_M = negative_prob_M / (torch.sum(negative_prob_M,
                dim=1) + self.eps).view(-1, 1).expand_as(negative_prob_M)

        if 'entropy_loss' in self.loss_type:
            #loss_en = - torch.mean(torch.sum(norm_neg_prob_M * torch.log(norm_neg_prob_M+
                #self.eps), dim=1))
            loss_en = - torch.mean(torch.sum(prediction * torch.log(prediction +
                self.eps), dim=1))
            loss += loss_en * self.loss_type.get('entropy_loss', 1)

        # loss to make sure all 
        loss_uniform = 0
        if 'uniform_loss' in self.loss_type:
            avg_norm_neg_prob_M = torch.mean(norm_neg_prob_M, dim=0)
            loss_uniform = -torch.mean(torch.log(avg_norm_neg_prob_M +
                self.eps)) - Variable(torch.log(torch.FloatTensor([self.M]).cuda()))
            #loss_uniform *= Variable(torch.FloatTensor([0.001]).cuda())
            loss += loss_uniform * self.loss_type.get('uniform_loss', 1)
            if (self._iter % 100) == 0:
                logging.info('loss ce = {}; loss en = {}; loss uniform = {}'.format(
                            loss_ce.data.cpu()[0], 
                            loss_en.data.cpu()[0], 
                            loss_uniform.data.cpu()[0]))

        if 'max_out' in self.loss_type:
            pred_N = pred.index_select(1, 
                    torch.autograd.Variable(torch.arange(0, self.N).long().cuda()))
            pred_M = pred.index_select(1,
                    torch.autograd.Variable(torch.arange(self.N, self.N +
                        self.M).long().cuda()))
            pred_maxM, _ = torch.max(pred_M, dim=1, keepdim=True)
            pred_NmaxM = torch.cat((pred_N, pred_maxM), dim=1)
            loss += self._ce(pred_NmaxM, target)

        self._iter = self._iter + 1
        return loss 
        
class MNet(nn.Module):
    def __init__(self, num_feature, N, M, loss_type):
        super(MNet, self).__init__()
        self.fc1 = nn.Linear(num_feature, N + M, bias=False)
        self.softmax = nn.Softmax()
        self.loss = NMLoss(N, M, loss_type)

    def forward(self, x):
        return self.fc1(x)

def train_model(trainloader, net, criterion, epochs=50):
    net.cuda()
    optimizer = optim.SGD(net.parameters(), lr=0.1, momentum=0.9)
    for epoch in range(epochs):  # loop over the dataset multiple times
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # get the inputs
            inputs, labels = data
            inputs = inputs.cuda()
            labels = labels.cuda()
            # wrap them in Variable
            inputs = Variable(inputs)
            labels = Variable(labels)
    
            # zero the parameter gradients
            optimizer.zero_grad()
    
            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
    
            # print statistics
            running_loss += loss.data[0]
            if (i % 200) == 0:    # print every 2000 mini-batches
                logging.info('[%d, %5d] loss: %.3f' %
                      (epoch, i, running_loss / 200.))
                running_loss = 0.0

def get_pca(x, com):
    x -= np.mean(x, axis = 0)
    cov = np.cov(x, rowvar=False)
    evals , evecs = LA.eigh(cov)
    idx = np.argsort(evals)[::-1]
    evecs = evecs[:,idx]
    evals = evals[idx]
    a = np.dot(x, evecs[:, :2])
    return a

def check_msoft_mnist():
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    torch.manual_seed(777)
    save_path = './output/pt_mnist/'
    train_feature_file = op.join(save_path, 'train_feature.pt')
    train_target_file = op.join(save_path, 'train_target.pt')

    t_train_features = torch.load(train_feature_file).cpu()
    t_train_labels = torch.load(train_target_file).cpu()

    train_features = t_train_features.numpy()
    train_labels = t_train_labels.numpy()

    test_feature_file = op.join(save_path, 'test_feature.pt')
    test_target_file = op.join(save_path, 'test_target.pt')

    t_test_features = torch.load(test_feature_file).cpu()
    t_test_labels = torch.load(test_target_file).cpu()

    num_real_classes = 10
    num_classes = 2 # the first 2 classes
    num_train_sample = t_train_features.shape[0]
    num_test_sample = t_test_features.shape[0]
    feature_dim = t_train_features.shape[1]
    radius = 10
    sigma = 1
    batch = 100
    epochs = 1

    # construct the testing data
    testset = torch.utils.data.TensorDataset(t_test_features, t_test_labels)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=1,
                                              shuffle=False, num_workers=2)
    trainset = torch.utils.data.TensorDataset(t_train_features, t_train_labels)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch,
                                              shuffle=True, num_workers=2)
    net = Net(feature_dim, num_real_classes)
    net.cuda()
    criterion = nn.CrossEntropyLoss()
    train_model(train_loader, net, criterion, epochs)
    acc_full_train = model_test(test_loader, net, num_real_classes)
    acc_full = model_test(test_loader, net, num_real_classes)
    logging.info('Full: acc on train = {}; acc on test = {}'.format(
        acc_full_train, acc_full))

    # simulate the training data with num_classes and all the rest is the
    # background
    t_train_labels[t_train_labels >= num_classes] = num_classes
    t_test_labels[t_test_labels >= num_classes] = num_classes

    trainset = torch.utils.data.TensorDataset(t_train_features, t_train_labels)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch,
                                              shuffle=True, num_workers=2)
    # if we only use num_classes + 1 entry for softmax 
    net = Net(feature_dim, num_classes + 1)
    train_model(train_loader, net, criterion, epochs)
    acc_n1_train = model_test(train_loader, net, num_classes)
    acc_n1_test = model_test(test_loader, net, num_classes)
    logging.info('n1 base: train = {}; test = {}'.format(acc_n1_train,
        acc_n1_test))
    
    # N + M approach - only cross entropy
    net = MNet(feature_dim, num_classes, 
            num_real_classes-num_classes, 
            loss_type=['cross_entropy'])
    criterion = net.loss
    train_model(train_loader, net, criterion, epochs)
    acc_nm_ce_train = model_test(train_loader, net, num_classes)
    acc_nm_ce_test = model_test(test_loader, net, num_classes)
    logging.info('N+M, CE; train = {}; test = {}'.format(acc_nm_ce_train, 
        acc_nm_ce_test))

    # N + M approach - ce + entropy
    net = MNet(feature_dim, num_classes, 
            num_real_classes-num_classes, 
            loss_type=['cross_entropy', 'entropy_loss'])
    criterion = net.loss
    train_model(train_loader, net, criterion, epochs)
    acc_nm_ceen_train = model_test(train_loader, net, num_classes)
    acc_nm_ceen_test = model_test(test_loader, net, num_classes)
    logging.info('N+M; CE+EN; train = {}; test = {}'.format(acc_nm_ceen_train, 
        acc_nm_ceen_test))

    # N + M approach, ce + entropy + uniform
    net = MNet(feature_dim, num_classes, 
            num_real_classes-num_classes, 
            loss_type=['cross_entropy', 'entropy_loss', 'uniform_loss'])
    criterion = net.loss
    train_model(train_loader, net, criterion, epochs)
    acc_nm_full_train = model_test(train_loader, net, num_classes)
    acc_nm_full_test = model_test(test_loader, net, num_classes)
    logging.info('N+M, full; {}-{}'.format(acc_nm_full_train, 
        acc_nm_full_test))

def main_entry():
    #test_random_data()
    #train_base_on_mnist()
    #check_msoft_mnist()
    #extract_feature_from_n1()
    #visualize_features('n1_train_feature.pt',
            #'n1_train_target.pt', 
            #'visualize_feat_n1')
    #extract_feature_from_nm()
    #visualize_features('nm_ce_train_feature.pt',
            #'nm_ce_train_target.pt',
            #'visualize_feat_nm')
    #mnist_full_train()
    #save_path = './output/pt_mnist/'
    #w_fname = op.join(save_path, 'nm_w.pt')
    #w = torch.load(w_fname)
    #w2 = torch.sum(w * w, dim=1)
    #logging.info(','.join(w2))
    test_single_layer()

def test_single_layer(): 
    train_feature_name = 'nm_ce_train_feature.pt'
    train_target_name = 'nm_ce_train_target.pt'
    test_feature_name = 'nm_ce_test_feature.pt'
    test_target_name = 'nm_ce_test_target.pt'

    train_feature_name = 'n1_train_feature.pt'
    train_target_name = 'n1_train_target.pt'
    test_feature_name = 'n1_test_feature.pt'
    test_target_name = 'n1_test_target.pt'

    save_path = './output/pt_mnist/'
    batch_size = 100
    num_classes = 2
    train_epochs = 10
    background_classes = 8

    train_feature_fname = op.join(save_path, train_feature_name)
    train_feature = torch.load(train_feature_fname)
    train_target_fname = op.join(save_path, train_target_name)
    train_target = torch.load(train_target_fname)
    trainset = torch.utils.data.TensorDataset(train_feature, train_target)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
            shuffle=True, num_workers=2)

    test_feature_fname = op.join(save_path, test_feature_name)
    test_feature = torch.load(test_feature_fname)
    test_target_fname = op.join(save_path, test_target_name)
    test_target = torch.load(test_target_fname)
    testset = torch.utils.data.TensorDataset(test_feature, test_target)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
            shuffle=False, num_workers=2)

    result = single_linear(train_loader, test_loader, num_classes, train_epochs,
            background_classes)

    logging.info(pformat(result))


def extract_feature_from_nm():
    with_cuda = True
    batch_size = 100
    num_classes = 2 # == N
    train_epochs = 10
    test_batch_size = 100
    background_classes = 8

    kwargs = {'num_workers': 1, 'pin_memory': True} if with_cuda else {}
    train_dataset = datasets.MNIST('../data', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ]))
    test_dataset = datasets.MNIST('../data', train=False, transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ]))
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=test_batch_size, shuffle=True, **kwargs)
    
    # mark the rest M as background
    train_labels = train_dataset.train_labels
    train_labels[train_labels >= num_classes] = num_classes
    test_labels = test_dataset.test_labels
    test_labels[test_labels >= num_classes] = num_classes
    
    # learning on N + 1
    net = MNISTMNet2(num_classes, 
            background_classes, 
            loss_type=['cross_entropy'])
    net.cuda()
    MNISTMtrain(train_loader, net, train_epochs)

    save_path = './output/pt_mnist/'
    test_feature_file = op.join(save_path, 'nm_ce_test_feature.pt')
    test_target_file = op.join(save_path, 'nm_ce_test_target.pt')
    MNISTextract_feature(test_loader, net, test_feature_file, 
            test_target_file)

    save_path = './output/pt_mnist/'
    train_feature_file = op.join(save_path, 'nm_ce_train_feature.pt')
    train_target_file = op.join(save_path, 'nm_ce_train_target.pt')
    MNISTextract_feature(train_loader, net, train_feature_file, 
            train_target_file)

    all_param = list(net.parameters())
    all_param[-1].data
    w_file = op.join(save_path, 'nm_w.pt')
    torch.save(all_param[-1].data, w_file)
    
def visualize_features(train_feature_name,
        train_target_name, save_file_name):
    from sklearn.manifold import TSNE
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    torch.manual_seed(777)
    save_path = './output/pt_mnist/'
    train_feature_file = op.join(save_path, train_feature_name)
    train_target_file = op.join(save_path, train_target_name)

    train_features = torch.load(train_feature_file).cpu().numpy()[:500]
    train_labels = torch.load(train_target_file).cpu().numpy()[:500]

    tsne = TSNE(n_components=2, random_state=0)
    #f2d = tsne.fit_transform(np.vstack((train_features, w)))
    #train_features_2d = f2d[:len(train_features)]
    train_features_2d = tsne.fit_transform(train_features)
    unique_labels = np.unique(train_labels)
    colors = gen_colors(1000)
    for l in unique_labels:
        plt.scatter(train_features_2d[train_labels == l, 0],
                train_features_2d[train_labels == l, 1],
                label=l, c=colors[l])
    plt.legend()
    #for i, (x, y) in enumerate(w_2d):
        #plt.plot([0, x], [0, y], color=colors[i])
    plt.grid()
    plt.savefig('/home/jianfw/work/jianfw_desktop/MSoftmax/{}.eps'.format(
        save_file_name), format='eps', bbox_inches='tight')
    plt.show()


def extract_feature_from_n1():
    with_cuda = True
    batch_size = 100
    num_classes = 2 # == N
    train_epochs = 10
    test_batch_size = 100
    background_classes = 8

    kwargs = {'num_workers': 1, 'pin_memory': True} if with_cuda else {}
    train_dataset = datasets.MNIST('../data', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ]))
    test_dataset = datasets.MNIST('../data', train=False, transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ]))
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=test_batch_size, shuffle=True, **kwargs)
    
    # mark the rest M as background
    train_labels = train_dataset.train_labels
    train_labels[train_labels >= num_classes] = num_classes
    test_labels = test_dataset.test_labels
    test_labels[test_labels >= num_classes] = num_classes
    
    # learning on N + 1
    net = MNISTNet(num_classes + 1)
    net.cuda()
    MNISTtrain(train_loader, net, train_epochs)
    acc_n1_train = MNISTtest(train_loader, net)
    acc_n1_test = MNISTtest(test_loader, net)
    logging.info('N+1: train = {}; test = {}'.format(acc_n1_train, 
        acc_n1_test))

    save_path = './output/pt_mnist/'
    test_feature_file = op.join(save_path, 'n1_test_feature.pt')
    test_target_file = op.join(save_path, 'n1_test_target.pt')
    MNISTextract_feature(test_loader, net, test_feature_file, 
            test_target_file)

    save_path = './output/pt_mnist/'
    train_feature_file = op.join(save_path, 'n1_train_feature.pt')
    train_target_file = op.join(save_path, 'n1_train_target.pt')
    MNISTextract_feature(train_loader, net, train_feature_file, 
            train_target_file)

    all_param = list(net.parameters())
    all_param[-1].data
    w_file = op.join(save_path, 'n1_w.pt')
    torch.save(all_param[-1].data, w_file)

def mnist_full_train():
    with_cuda = True
    batch_size = 100
    num_classes = 2 # == N
    train_epochs = 10
    test_batch_size = 100
    background_classes = 8

    kwargs = {'num_workers': 1, 'pin_memory': True} if with_cuda else {}
    train_dataset = datasets.MNIST('../data', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ]))
    test_dataset = datasets.MNIST('../data', train=False, transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ]))
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=test_batch_size, shuffle=True, **kwargs)
    
    # mark the rest M as background
    train_labels = train_dataset.train_labels
    train_labels[train_labels >= num_classes] = num_classes
    test_labels = test_dataset.test_labels
    test_labels[test_labels >= num_classes] = num_classes
    
    # learning on N + 1
    net = MNISTNet(num_classes + 1)
    net.cuda()
    MNISTtrain(train_loader, net, train_epochs)
    acc_n1_train = MNISTtest(train_loader, net)
    acc_n1_test = MNISTtest(test_loader, net)
    logging.info('N+1: train = {}; test = {}'.format(acc_n1_train, 
        acc_n1_test))

    # learning on N + M
    ce = ['cross_entropy']
    ce_en = ['cross_entropy', 'entropy_loss']
    ce_en_uniform = ['cross_entropy', 'entropy_loss', 'uniform_loss']
    #for loss_type in [ce, ce_en, ce_en_uniform]:
    for loss_type in []:
        net = MNISTMNet(num_classes, 
                background_classes, 
                loss_type=loss_type)
        net.cuda()
        MNISTMtrain(train_loader, net, train_epochs)
        acc_nm_train = model_test(train_loader, net, num_classes)
        acc_nm_test = model_test(test_loader, net, num_classes)
        logging.info('N + M: {}: train = {}; test = {}'.format(
            '.'.join(loss_type),
            acc_nm_train, acc_nm_test))

class MNISTNet(nn.Module):
    def __init__(self, num_classes=10):
        super(MNISTNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, num_classes, bias=False)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        feature = F.dropout(x, training=self.training)
        x = self.fc2(feature)
        return F.log_softmax(x, dim=1), feature

class MNISTMNet(nn.Module):
    def __init__(self, N, M, loss_type):
        super(MNISTMNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, N + M)
        self.softmax = nn.Softmax()
        self.loss = NMLoss(N, M, loss_type)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        feature = F.dropout(x, training=self.training)
        return self.fc2(feature)

class MNISTMNet2(nn.Module):
    def __init__(self, N, M, loss_type):
        super(MNISTMNet2, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, N + M, bias=False)
        self.softmax = nn.Softmax()
        self.loss = NMLoss(N, M, loss_type)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        feature = F.dropout(x, training=self.training)
        return self.fc2(feature), feature

def MNISTMtrain(train_loader, model, epochs):
    with_cuda = torch.cuda.is_available()
    lr = 0.01
    momentum = 0.5
    model.train()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)
    for epoch in range(epochs):
        for batch_idx, (data, target) in enumerate(train_loader):
            if with_cuda:
                data, target = data.cuda(), target.cuda()
            data, target = Variable(data), Variable(target)
            optimizer.zero_grad()
            output = model(data)
            if type(output) is tuple:
                output = output[0]
            loss = model.loss(output, target)
            if type(loss) is tuple:
                loss = loss[0]
            loss.backward()
            optimizer.step()
            #if batch_idx % 100 == 0:
                #logging.info('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    #epoch, batch_idx * len(data), len(train_loader.dataset),
                    #100. * batch_idx / len(train_loader), loss.data[0]))
    
def MNISTtrain(train_loader, model, epochs):
    with_cuda = torch.cuda.is_available()
    lr = 0.01
    momentum = 0.5
    model.train()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)
    for epoch in range(epochs):
        for batch_idx, (data, target) in enumerate(train_loader):
            if with_cuda:
                data, target = data.cuda(), target.cuda()
            data, target = Variable(data), Variable(target)
            optimizer.zero_grad()
            output, _ = model(data)
            loss = F.nll_loss(output, target)
            loss.backward()
            optimizer.step()
            if batch_idx % 100 == 0:
                logging.info('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader), loss.data[0]))

def MNISTtest(test_loader, model):
    model.eval()
    test_loss = 0
    correct = 0
    for data, target in test_loader:
        data, target = data.cuda(), target.cuda()
        data, target = Variable(data, volatile=True), Variable(target)
        output, _ = model(data)
        test_loss += F.nll_loss(output, target, size_average=False).data[0] # sum up batch loss
        pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
        correct += pred.eq(target.data.view_as(pred)).long().cpu().sum()

    test_loss /= len(test_loader.dataset)
    logging.info('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    return 100.0 * correct / len(test_loader.dataset)

def MNISTextract(test_loader, model):
    model.eval()
    test_loss = 0
    correct = 0
    for data, target in test_loader:
        data, target = data.cuda(), target.cuda()
        v_data = Variable(data, volatile=True)
        output, feature = model(v_data)
        yield feature, target

def MNISTextract_feature(data_loader, model, feature_file, target_file):
    # extract the feature
    all_feature = []
    all_target = []
    model.eval()
    for feature, target in MNISTextract(data_loader, model):
        all_feature.append(feature)
        all_target.append(target)
    feature = torch.cat(all_feature)
    target = torch.cat(all_target)
    ensure_directory(op.dirname(feature_file))
    torch.save(feature.data.cpu(), feature_file)
    ensure_directory(op.dirname(target_file))
    torch.save(target.cpu(), target_file)

def train_base_on_mnist():
    with_cuda = torch.cuda.is_available()
    seed = 1
    batch_size = 64
    test_batch_size = 1000
    log_interval = 10
    epochs = 10
    epochs = 1
    
    kwargs = {'num_workers': 1, 'pin_memory': True} if with_cuda else {}
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=False, transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=test_batch_size, shuffle=True, **kwargs)
    
    model = MNISTNet()
    model.cuda()
    
    for epoch in range(1, epochs + 1):
        MNISTtrain(train_loader, model, epoch)
        MNISTtest(test_loader, model)
    save_path = './output/pt_mnist/'
    feature_file = op.join(save_path, 'train_feature.pt')
    target_file = op.join(save_path, 'train_target.pt')
    MNISTextract_feature(train_loader, model, feature_file, 
            target_file)
    feature_file = op.join(save_path, 'test_feature.pt')
    target_file = op.join(save_path, 'test_target.pt')
    MNISTextract_feature(test_loader, model, feature_file, 
            target_file)

def shrink_M_to_1(output, loss_type, N, num_proto, 
        multi_policy_proto):
    batch_size = output.shape[0]
    output_dim = output.data.shape[1]
    if 'max_out' in loss_type:
        first_pos = output.index_select(1, 
                Variable(torch.arange(0, N).long().cuda()))
        second_neg = output.index_select(1,
                Variable(torch.arange(
                    N, output_dim).long().cuda()))
        second_max_neg, _ = torch.max(second_neg, dim=1, keepdim=True)
        output = torch.cat((first_pos, second_max_neg), dim=1)
    else:
        if num_proto > 1:
            if multi_policy_proto == 'max_softmax':
                first = output[:, :(N * num_proto)].contiguous().view(
                        batch_size, N, num_proto)
                first_max, _ = torch.max(first, dim=2)
                second = output[:, (N * num_proto): ]
                pred = torch.cat((first_max, second), dim=1)
                output = F.softmax(pred, dim=1)
            else:
                assert False

        output = F.softmax(output, dim=1)
        first_pos = output.index_select(1, 
                Variable(torch.arange(0, N).long().cuda()))
        second_neg = output.index_select(1,
                Variable(torch.arange(
                    N, output_dim).long().cuda()))
        prob_neg = torch.sum(second_neg, 1, keepdim=True)
        output = torch.cat((first_pos, prob_neg), dim=1)
    return output

def model_test(test_loader, net, num_pos_classes, loss_type=[]):
    net.eval()
    with_cuda = True
    correct = 0
    for inputs, labels in test_loader:
        if with_cuda:
            inputs, labels = inputs.cuda(), labels.cuda()
        v_inputs = Variable(inputs)
        v_labels = Variable(labels)
        output = net(v_inputs)
        output_dim = output.data.shape[1]
        if output_dim != num_pos_classes:
            if 'max_out' in loss_type:
                '''
                deprecated. call shrink_M_to_1
                '''
                first_pos = output.index_select(1, 
                        Variable(torch.arange(0, num_pos_classes).long().cuda()))
                second_neg = output.index_select(1,
                        Variable(torch.arange(
                            num_pos_classes, output_dim).long().cuda()))
                second_max_neg, _ = torch.max(second_neg, dim=1, keepdim=True)
                output = torch.cat((first_pos, second_max_neg), dim=1)
            else:
                output = F.softmax(output, dim=1)
                first_pos = output.index_select(1, 
                        Variable(torch.arange(0, num_pos_classes).long().cuda()))
                second_neg = output.index_select(1,
                        Variable(torch.arange(
                            num_pos_classes, output_dim).long().cuda()))
                prob_neg = torch.sum(second_neg, 1, keepdim=True)
                output = torch.cat((first_pos, prob_neg), dim=1)

        pred_idx = output.data.max(1, keepdim=True)[1]
        correct += pred_idx.eq(
                labels.view_as(pred_idx)).long().cpu().sum()

    return 1. * correct / len(test_loader.dataset)

def test_random_data():
    test_random_data_one(background_classes=8, train_epochs=10)
    test_random_data_one(background_classes=10, train_epochs=10)
    test_random_data_one(background_classes=6, train_epochs=10)

def single_linear(train_loader, test_loader, 
        num_classes, train_epochs, background_classes):
    # if we only use num_classes + 1 entry for softmax 
    result = {}
    assert len(train_loader.dataset.data_tensor.shape) == 2
    feature_dim = train_loader.dataset.data_tensor.shape[1]

    criterion = nn.CrossEntropyLoss()
    net = Net(feature_dim, num_classes + 1)
    train_model(train_loader, net, criterion, train_epochs)
    acc_n1_train = model_test(train_loader, net, num_classes)
    acc_n1_test = model_test(test_loader, net, num_classes)
    result['acc_n1'] = (acc_n1_train, acc_n1_test)
    logging.info('n1 base: train = {}; test = {}'.format(acc_n1_train,
        acc_n1_test))
    
    ce = ['cross_entropy']
    ce_en = ['cross_entropy', 'entropy_loss']
    ce_en_uniform = ['cross_entropy', 'entropy_loss', 'uniform_loss']
    max_out = ['max_out']
    for loss_type in [ce, ce_en, ce_en_uniform, max_out]:
    #for loss_type in [max_out]:
        # N + M approach - only cross entropy
        net = MNet(feature_dim, num_classes, 
                background_classes, 
                loss_type=loss_type)
        criterion = net.loss
        train_model(train_loader, net, criterion, train_epochs)
        acc_train = model_test(train_loader, net, num_classes, loss_type)
        acc_test = model_test(test_loader, net, num_classes, loss_type)
        logging.info('{} - {} - {}'.format(','.join(loss_type),
            acc_train, acc_test))
        result[','.join(loss_type)] = (acc_train, acc_test)

    return result


def test_random_data_one(background_classes, train_epochs):
    init_logging()
    torch.manual_seed(777)
    num_real_classes = 10
    num_classes = 2 # the first 2 classes
    num_train_sample = 1000
    num_test_sample = 100
    radius = 10
    sigma = 1
    batch = 64
    class_grad = range(num_real_classes)
    #t = class_grad[1]
    #class_grad[1] = class_grad[2]
    #class_grad[2] = t
    train_features, train_labels = get_data(num_real_classes, num_train_sample,
            radius, sigma, class_grad)
    test_features, test_labels = get_data(num_real_classes, num_test_sample,
            radius, sigma, class_grad)

    # construct the testing data
    t_test_features = torch.from_numpy(test_features).float()
    t_test_labels = torch.from_numpy(test_labels).long()
    testset = torch.utils.data.TensorDataset(t_test_features, t_test_labels)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=1,
                                              shuffle=False, num_workers=2)
    # train
    t_train_features = torch.from_numpy(train_features).float()
    t_train_labels = torch.from_numpy(train_labels).long()
    trainset = torch.utils.data.TensorDataset(t_train_features, t_train_labels)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch,
                                              shuffle=True, num_workers=2)
    _, axs = plt.subplots(2, 3)
    net = Net(2, num_real_classes)
    criterion = nn.CrossEntropyLoss()
    train_model(train_loader, net, criterion, train_epochs)
    acc_full_train = model_test(test_loader, net, num_real_classes)
    acc_full = model_test(test_loader, net, num_real_classes)
    logging.info('Full: acc on train = {}; acc on test = {}'.format(
        acc_full_train, acc_full))
    colors = gen_colors(1000)
    plot(net, train_features, train_labels, colors, radius, axs[0, 0])
    axs[0, 0].set_title('{} classes, ce, {}-{}'.format(num_real_classes,
        acc_full_train, acc_full))
    
    # simulate the training data with num_classes and all the rest is the
    # background
    t_train_labels[t_train_labels >= num_classes] = num_classes
    t_test_labels[t_test_labels >= num_classes] = num_classes

    trainset = torch.utils.data.TensorDataset(t_train_features, t_train_labels)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch,
                                              shuffle=True, num_workers=2)
    # if we only use num_classes + 1 entry for softmax 
    net = Net(2, num_classes + 1)
    train_model(train_loader, net, criterion, train_epochs)
    acc_n1_train = model_test(train_loader, net, num_classes)
    acc_n1_test = model_test(test_loader, net, num_classes)
    logging.info('n1 base: train = {}; test = {}'.format(acc_n1_train,
        acc_n1_test))
    plot(net, train_features, train_labels, colors, radius, axs[1, 0])
    axs[1, 0].set_title('{} + 1 classes, ce {}-{}'.format(
        num_classes, acc_n1_train, acc_n1_test))
    
    # N + M approach - only cross entropy
    net = MNet(2, num_classes, 
            background_classes, 
            loss_type=['cross_entropy'])
    criterion = net.loss
    train_model(train_loader, net, criterion, train_epochs)
    acc_nm_ce_train = model_test(train_loader, net, num_classes)
    acc_nm_ce_test = model_test(test_loader, net, num_classes)
    logging.info('N+M, CE; train = {}; test = {}'.format(acc_nm_ce_train, 
        acc_nm_ce_test))
    plot(net, train_features, train_labels, colors, radius, axs[0, 1])
    axs[0, 1].set_title('N + M, ce; {} - {}'.format(acc_nm_ce_train,
        acc_nm_ce_test))

    # N + M approach - ce + entropy
    net = MNet(2, num_classes, 
            background_classes, 
            loss_type=['cross_entropy', 'entropy_loss'])
    criterion = net.loss
    train_model(train_loader, net, criterion, train_epochs)
    acc_nm_ceen_train = model_test(train_loader, net, num_classes)
    acc_nm_ceen_test = model_test(test_loader, net, num_classes)
    logging.info('N+M; CE+EN; train = {}; test = {}'.format(acc_nm_ceen_train, 
        acc_nm_ceen_test))
    plot(net, train_features, train_labels, colors, radius, axs[1, 1])
    axs[1, 1].set_title('N + M, ce + en; {}-{}'.format(
        acc_nm_ceen_train, acc_nm_ceen_test))

    # N + M approach, ce + entropy + uniform
    net = MNet(2, num_classes, 
            background_classes, 
            loss_type=['cross_entropy', 'entropy_loss', 'uniform_loss'])
    criterion = net.loss
    train_model(train_loader, net, criterion, train_epochs)
    acc_nm_full_train = model_test(train_loader, net, num_classes)
    acc_nm_full_test = model_test(test_loader, net, num_classes)
    logging.info('N+M, full; {}-{}'.format(acc_nm_full_train, 
        acc_nm_full_test))
    plot(net, train_features, train_labels, colors, radius, axs[0, 2])
    axs[0, 2].set_title('N + M, ce + en + uniform; {}-{}'.format(
        acc_nm_full_train, acc_nm_full_test))
    logging.info('start saving')

    fig = plt.gcf()
    #fig.set_figure_width = 8.5
    fig.set_size_inches((10.5, 7), forward=False)

    plt.savefig('/home/jianfw/work/jianfw_desktop/MSoftmax/random_{}.eps'.format(
        background_classes), format='eps', bbox_inches='tight')
    plt.show()

if __name__ == '__main__':
    init_logging()
    main_entry()
