from torch.autograd import Variable
from torch.utils.data import DataLoader
import os
import argparse
import pickle
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier
import time
import utils
from HSItrans_CLIP_Houston import *

# np.random.seed(1337)

parser = argparse.ArgumentParser(description="Few Shot Visual Recognition")
parser.add_argument("-f", "--feature_dim", type=int, default=128)
parser.add_argument("-c", "--src_input_dim", type=int, default=128)
parser.add_argument("-d", "--tar_input_dim", type=int, default=144)  # PaviaU=103；salinas=204
parser.add_argument("-n", "--n_dim", type=int, default=100)
parser.add_argument("-w", "--class_num", type=int, default=15)
parser.add_argument("-s", "--shot_num_per_class", type=int, default=1)
parser.add_argument("-b", "--query_num_per_class", type=int, default=19)
parser.add_argument("-e", "--episode", type=int, default=20000)
parser.add_argument("-t", "--test_episode", type=int, default=600)
parser.add_argument("-l", "--learning_rate", type=float, default=0.001)
parser.add_argument("-g", "--gpu", type=int, default=0)
parser.add_argument("-u", "--hidden_unit", type=int, default=10)
# target
parser.add_argument("-m", "--test_class_num", type=int, default=15)
parser.add_argument("-z", "--test_lsample_num_per_class", type=int, default=5, help='5 4 3 2 1')

args = parser.parse_args(args=[])

# Hyper Parameters
FEATURE_DIM = args.feature_dim
SRC_INPUT_DIMENSION = args.src_input_dim
TAR_INPUT_DIMENSION = args.tar_input_dim
N_DIMENSION = args.n_dim
CLASS_NUM = args.class_num
SHOT_NUM_PER_CLASS = args.shot_num_per_class
QUERY_NUM_PER_CLASS = args.query_num_per_class
EPISODE = args.episode
TEST_EPISODE = args.test_episode
LEARNING_RATE = args.learning_rate
GPU = args.gpu
HIDDEN_UNIT = args.hidden_unit

# Hyper Parameters in target domain data set
TEST_CLASS_NUM = args.test_class_num  # the number of class
TEST_LSAMPLE_NUM_PER_CLASS = args.test_lsample_num_per_class  # the number of labeled samples per class 5 4 3 2 1

utils.same_seeds(0)


def _init_():
    if not os.path.exists('checkpoints'):
        os.makedirs('checkpoints')
    if not os.path.exists('classificationMap'):
        os.makedirs('classificationMap')


_init_()

# load source domain data set
# with open(os.path.join('/data/haodong1_backup/datasets', 'Chikusei_imdb_128.pickle'), 'rb') as handle:
with open(os.path.join('datasets', 'Chikusei_imdb_128.pickle'), 'rb') as handle:
    source_imdb = pickle.load(handle)
print(source_imdb.keys())
print(source_imdb['Labels'])

# process source domain data set
data_train = source_imdb['data']  # (77592, 9, 9, 128)
labels_train = source_imdb['Labels']  # 77592
print(data_train.shape)
print(labels_train.shape)
keys_all_train = sorted(list(set(labels_train)))  # class [0,...,18]
print(keys_all_train)  # [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18]
label_encoder_train = {}
for i in range(len(keys_all_train)):
    label_encoder_train[keys_all_train[i]] = i
print(label_encoder_train)

train_set = {}
for class_, path in zip(labels_train, data_train):
    if label_encoder_train[class_] not in train_set:
        train_set[label_encoder_train[class_]] = []
    train_set[label_encoder_train[class_]].append(path)
print(train_set.keys())
data = train_set
del train_set
del keys_all_train
del label_encoder_train

print("Num classes for source domain datasets: " + str(len(data)))
print(data.keys())
data = utils.sanity_check(data)  # 200 labels samples per class
print("Num classes of the number of class larger than 200: " + str(len(data)))

for class_ in data:
    for i in range(len(data[class_])):
        image_transpose = np.transpose(data[class_][i], (2, 0, 1))  # （9,9,100）-> (100,9,9)
        data[class_][i] = image_transpose

# source few-shot classification data
metatrain_data = data
print(len(metatrain_data.keys()), metatrain_data.keys())
del data

# source domain adaptation data
print(source_imdb['data'].shape)  # (77592, 9, 9, 100)
source_imdb['data'] = source_imdb['data'].transpose((1, 2, 3, 0))  # (9, 9, 100, 77592)
print(source_imdb['data'].shape)  # (77592, 9, 9, 100)
print(source_imdb['Labels'])
source_dataset = utils.matcifar(source_imdb, train=True, d=3, medicinal=0)
source_loader = torch.utils.data.DataLoader(source_dataset, batch_size=128, shuffle=True, num_workers=0)
del source_dataset, source_imdb

## target domain data set
# load target domain data set
test_data = '/data/HSI-newtestData/Houston_data.mat'
test_label = '/data/HSI-newtestData/Houston_gt.mat'

Data_Band_Scaler, GroundTruth = utils.load_data(test_data, test_label)


# get train_loader and test_loader
def get_train_test_loader(Data_Band_Scaler, GroundTruth, class_num, shot_num_per_class):
    print(Data_Band_Scaler.shape)  # (610, 340, 103)
    [nRow, nColumn, nBand] = Data_Band_Scaler.shape
    label_values = ["Water【Natural】【medium density】【high specific heat capability】",
                    "Bare soil (school)【Agricultural】【low density】【medium specific heat capability】",
                    "Bare soil (park)【Agricultural】【low density】【medium specific heat capability】",
                    "Bare soil (farmland)【Agricultural】【low density】【medium specific heat capability】",
                    "Natural plants【Natural】【low density】【medium specific heat capability】",
                    "Weeds in farmland【Agricultural】【low density】【medium specific heat capability】",
                    "Forest【Natural】【low density】【medium specific heat capability】",
                    "Grass【Natural】【low density】【high specific heat capability】",
                    "Rice field (grown)【Agricultural】【low density】【high specific heat capability】",
                    "Rice field (first stage)【Agricultural】【low density】【high specific heat capability】",
                    "Row crops【Agricultural】【low density】【high specific heat capability】",
                    "Plastic house【Man-made】【high density】【high specific heat capability】",
                    "Manmade(non-dark)【Man-made】【high density】【high specific heat capability】",
                    "Manmade(dark)【Man-made】【high density】【high specific heat capability】",
                    "Manmade(blue)【Man-made】【high density】【high specific heat capability】",
                    "Manmade(red)【Man-made】【high density】【high specific heat capability】",
                    "Manmade grass【Man-made】【high density】【high specific heat capability】",
                    "Asphalt【Agricultural】【high density】【high specific heat capability】",
                    "Paved ground【Agricultural】【high density】【high specific heat capability】"
                    ]
    # label_values = ["Water",
    #                 "Bare soil (school)",
    #                 "Bare soil (park)",
    #                 "Bare soil (farmland)",
    #                 "Natural plants",
    #                 "Weeds in farmland",
    #                 "Forest",
    #                 "Grass",
    #                 "Rice field (grown)",
    #                 "Rice field (first stage)",
    #                 "Row crops",
    #                 "Plastic house",
    #                 "Manmade(non-dark)",
    #                 "Manmade(dark)",
    #                 "Manmade(blue)",
    #                 "Manmade(red)",
    #                 "Manmade grass",
    #                 "Asphalt",
    #                 "Paved ground"
    #                 ]
    '''label start'''
    num_class = int(np.max(GroundTruth))
    data_band_scaler = utils.flip(Data_Band_Scaler)
    groundtruth = utils.flip(GroundTruth)
    del Data_Band_Scaler
    del GroundTruth

    HalfWidth = 4
    G = groundtruth[nRow - HalfWidth:2 * nRow + HalfWidth, nColumn - HalfWidth:2 * nColumn + HalfWidth]
    data = data_band_scaler[nRow - HalfWidth:2 * nRow + HalfWidth, nColumn - HalfWidth:2 * nColumn + HalfWidth, :]

    [Row, Column] = np.nonzero(G)  # (10249,) (10249,)
    # print(Row)
    del data_band_scaler
    del groundtruth

    nSample = np.size(Row)
    print('number of sample', nSample)

    # Sampling samples
    train = {}
    test = {}
    da_train = {}  # Data Augmentation
    m = int(np.max(G))  # 9
    nlabeled = TEST_LSAMPLE_NUM_PER_CLASS
    print('labeled number per class:', nlabeled)
    print((200 - nlabeled) / nlabeled + 1)
    print(math.ceil((200 - nlabeled) / nlabeled) + 1)

    for i in range(m):
        indices = [j for j, x in enumerate(Row.ravel().tolist()) if G[Row[j], Column[j]] == i + 1]
        np.random.shuffle(indices)
        nb_val = shot_num_per_class
        train[i] = indices[:nb_val]
        da_train[i] = []
        for j in range(math.ceil((200 - nlabeled) / nlabeled) + 1):
            da_train[i] += indices[:nb_val]
        test[i] = indices[nb_val:]

    train_indices = []
    test_indices = []
    da_train_indices = []
    for i in range(m):
        train_indices += train[i]
        test_indices += test[i]
        da_train_indices += da_train[i]
    np.random.shuffle(test_indices)

    print('the number of train_indices:', len(train_indices))  # 520
    print('the number of test_indices:', len(test_indices))  # 9729
    print('the number of train_indices after data argumentation:', len(da_train_indices))  # 520
    print('labeled sample indices:', train_indices)

    nTrain = len(train_indices)
    nTest = len(test_indices)
    da_nTrain = len(da_train_indices)

    imdb = {}
    imdb['data'] = np.zeros([2 * HalfWidth + 1, 2 * HalfWidth + 1, nBand, nTrain + nTest],
                            dtype=np.float32)  # (9,9,100,n)
    imdb['Labels'] = np.zeros([nTrain + nTest], dtype=np.int64)
    imdb['set'] = np.zeros([nTrain + nTest], dtype=np.int64)

    RandPerm = train_indices + test_indices

    RandPerm = np.array(RandPerm)

    for iSample in range(nTrain + nTest):
        imdb['data'][:, :, :, iSample] = data[
                                         Row[RandPerm[iSample]] - HalfWidth:  Row[RandPerm[iSample]] + HalfWidth + 1,
                                         Column[RandPerm[iSample]] - HalfWidth: Column[
                                                                                    RandPerm[iSample]] + HalfWidth + 1,
                                         :]
        imdb['Labels'][iSample] = G[Row[RandPerm[iSample]], Column[RandPerm[iSample]]].astype(np.int64)

    imdb['Labels'] = imdb['Labels'] - 1  # 1-16 0-15
    imdb['set'] = np.hstack((np.ones([nTrain]), 3 * np.ones([nTest]))).astype(np.int64)
    print('Data is OK.')

    train_dataset = utils.matcifar(imdb, train=True, d=3, medicinal=0)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=class_num * shot_num_per_class, shuffle=False,
                                               num_workers=0)
    del train_dataset

    test_dataset = utils.matcifar(imdb, train=False, d=3, medicinal=0)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=100, shuffle=False, num_workers=0)
    del test_dataset
    del imdb

    # Data Augmentation for target domain for training
    imdb_da_train = {}
    imdb_da_train['data'] = np.zeros([2 * HalfWidth + 1, 2 * HalfWidth + 1, nBand, da_nTrain],
                                     dtype=np.float32)  # (9,9,100,n)
    imdb_da_train['Labels'] = np.zeros([da_nTrain], dtype=np.int64)
    imdb_da_train['set'] = np.zeros([da_nTrain], dtype=np.int64)

    da_RandPerm = np.array(da_train_indices)
    for iSample in range(da_nTrain):  # radiation_noise，flip_augmentation
        imdb_da_train['data'][:, :, :, iSample] = utils.radiation_noise(
            data[Row[da_RandPerm[iSample]] - HalfWidth:  Row[da_RandPerm[iSample]] + HalfWidth + 1,
            Column[da_RandPerm[iSample]] - HalfWidth: Column[da_RandPerm[iSample]] + HalfWidth + 1, :])
        imdb_da_train['Labels'][iSample] = G[Row[da_RandPerm[iSample]], Column[da_RandPerm[iSample]]].astype(np.int64)

    imdb_da_train['Labels'] = imdb_da_train['Labels'] - 1  # 1-16 0-15
    imdb_da_train['set'] = np.ones([da_nTrain]).astype(np.int64)
    print('ok')

    return train_loader, test_loader, imdb_da_train, G, RandPerm, Row, Column, nTrain, label_values


def get_target_dataset(Data_Band_Scaler, GroundTruth, class_num, shot_num_per_class):
    train_loader, test_loader, imdb_da_train, G, RandPerm, Row, Column, nTrain, labels_name_src = get_train_test_loader(
        Data_Band_Scaler=Data_Band_Scaler, GroundTruth=GroundTruth, \
        class_num=class_num, shot_num_per_class=shot_num_per_class)  # 9 classes and 5 labeled samples per class
    labels_name_tar = [
        "Healthy grass【Natural】【medium density】【high specific heat capability】",
        "Stressed grass【Natural】【medium density】【high specific heat capability】",
        "Synthetic grass【Natural】【medium density】【high specific heat capability】",
        "Trees【Natural】【medium density】【high specific heat capability】",
        "Soil【Natural】【medium density】【medium specific heat capability】",
        "Water【Natural】【medium density】【high specific heat capability】",
        "Residential buildings【Man-made】【high density】【low specific heat capability】",
        "Commercial buildings【Man-made】【high density】【low specific heat capability】",
        "Road【Man-made】【high density】【low specific heat capability】",
        "Highway【Man-made】【high density】【low specific heat capability】",
        "Railway【Man-made】【high density】【low specific heat capability】",
        "Parking Lot one【Man-made】【high density】【low specific heat capability】",
        "Parking Lot two【Man-made】【high density】【low specific heat capability】",
        "Tennis Court【Man-made】【high density】【low specific heat capability】",
        "Running Track【Man-made】【high density】【low specific heat capability】",
    ]
    # labels_name_tar = [
    #     "Healthy grass ",
    #     "Stressed grass",
    #     "Synthetic grass",
    #     "Trees",
    #     "Soil",
    #     "Water",
    #     "Residential buildings",
    #     "Commercial buildings",
    #     "Road",
    #     "Highway",
    #     "Railway",
    #     "Parking Lot one",
    #     "Parking Lot two",
    #     "Tennis Court",
    #     "Running Track",
    # ]
    train_datas, train_labels = train_loader.__iter__().__next__()
    print('train labels:', train_labels)
    print('train data label name:', labels_name_src)
    print('size of train datas:', train_datas.shape)  # size of train datas: torch.Size([45, 103, 9, 9])

    print(imdb_da_train.keys())
    print(imdb_da_train['data'].shape)  # (9, 9, 100, 225)
    print(imdb_da_train['Labels'])
    del Data_Band_Scaler, GroundTruth

    # target data with data augmentation
    target_da_datas = np.transpose(imdb_da_train['data'], (3, 2, 0, 1))  # (9,9,100, 1800)->(1800, 100, 9, 9)
    print(target_da_datas.shape)
    target_da_labels = imdb_da_train['Labels']  # (1800,)
    print('target data augmentation label:', target_da_labels)
    print('target data label name:', labels_name_tar)
    # metatrain data for few-shot classification
    target_da_train_set = {}
    for class_, path in zip(target_da_labels, target_da_datas):
        if class_ not in target_da_train_set:
            target_da_train_set[class_] = []
        target_da_train_set[class_].append(path)
    target_da_metatrain_data = target_da_train_set
    print(target_da_metatrain_data.keys())

    # target domain : batch samples for domian adaptation
    print(imdb_da_train['data'].shape)  # (9, 9, 100, 225)
    print(imdb_da_train['Labels'])
    target_dataset = utils.matcifar(imdb_da_train, train=True, d=3, medicinal=0)
    target_loader = torch.utils.data.DataLoader(target_dataset, batch_size=128, shuffle=True, num_workers=0)
    del target_dataset

    return train_loader, test_loader, target_da_metatrain_data, target_loader, G, RandPerm, Row, Column, nTrain,labels_name_src,labels_name_tar


crossEntropy = nn.CrossEntropyLoss().cuda()
domain_criterion = nn.BCEWithLogitsLoss().cuda()

# run 10 times
runs = 10
acc = np.zeros([runs, 1])
A = np.zeros([runs, CLASS_NUM])
k = np.zeros([runs, 1])
best_predict_all = []
best_acc_all = 0.0
best_G, best_RandPerm, best_Row, best_Column, best_nTrain = None, None, None, None, None

seeds = [1330, 1220, 1336, 1337, 1224, 1236, 1226, 1235, 1233, 1229]
for run in range(runs):
    print(run)
    # load target domain data for training and testing
    np.random.seed(seeds[run])
    train_loader, test_loader, target_da_metatrain_data, target_loader, G, RandPerm, Row, Column, nTrain ,labels_name_src,labels_name_tar= get_target_dataset(
        Data_Band_Scaler=Data_Band_Scaler, GroundTruth=GroundTruth, class_num=TEST_CLASS_NUM,
        shot_num_per_class=TEST_LSAMPLE_NUM_PER_CLASS)
    # model
    feature_encoder = TransformNet(output_unit=CLASS_NUM)
    domain_classifier = DomainClassifier()
    random_layer = RandomLayer([args.feature_dim*25, args.class_num], 1024)
    domain_classifier_text = DomainClassifier()

    feature_encoder.apply(weights_init)
    domain_classifier.apply(weights_init)
    domain_classifier_text.apply(weights_init)

    feature_encoder.cuda()
    domain_classifier.cuda()
    domain_classifier_text.cuda()
    random_layer.cuda()  # Random layer

    feature_encoder.train()
    domain_classifier.train()
    # optimizer
    feature_encoder_optim = torch.optim.Adam(feature_encoder.parameters(), lr=args.learning_rate)
    domain_classifier_optim = torch.optim.Adam(domain_classifier.parameters(), lr=args.learning_rate)
    domain_classifier_optim_text = torch.optim.Adam(domain_classifier_text.parameters(), lr=args.learning_rate)
    print("Training...")

    last_accuracy = 0.0
    best_episdoe = 0
    train_loss = []
    test_acc = []
    running_D_loss, running_F_loss = 0.0, 0.0
    running_label_loss = 0
    running_domain_loss = 0
    total_hit, total_num = 0.0, 0.0
    test_acc_list = []

    source_iter = iter(source_loader)
    target_iter = iter(target_loader)
    len_dataloader = min(len(source_loader), len(target_loader))
    train_start = time.time()
    for episode in range(10000):  # EPISODE = 90000
        # get domain adaptation data from  source domain and target domain
        try:
            source_data, source_label = source_iter.__next__()
        except Exception as err:
            source_iter = iter(source_loader)
            source_data, source_label = source_iter.__next__()

        try:
            target_data, target_label = target_iter.__next__()
        except Exception as err:
            target_iter = iter(target_loader)
            target_data, target_label = target_iter.__next__()

        # source domain few-shot + domain adaptation
        if episode % 2 == 0:
            '''Few-shot claification for source domain data set'''
            # get few-shot classification samples
            task = utils.Task(metatrain_data, CLASS_NUM, SHOT_NUM_PER_CLASS, QUERY_NUM_PER_CLASS)  # 5， 1，15
            support_dataloader = utils.get_HBKC_data_loader(task, num_per_class=SHOT_NUM_PER_CLASS, split="train",
                                                            shuffle=False)
            query_dataloader = utils.get_HBKC_data_loader(task, num_per_class=QUERY_NUM_PER_CLASS, split="test",
                                                          shuffle=True)
            text_dataloder = utils.get_HBKC_data_loader(task, num_per_class=1, split="train", shuffle=False)
            if episode == 0:
                task_tar = utils.Task(target_da_metatrain_data, TEST_CLASS_NUM, SHOT_NUM_PER_CLASS,
                                  QUERY_NUM_PER_CLASS)  # 5， 1，15
                text_dataloader = utils.get_HBKC_data_loader(task, num_per_class=1, split="train", shuffle=False)
                text_dataloader = utils.get_HBKC_data_loader(task, num_per_class=1, split="train", shuffle=False)
                _, text_labels = text_dataloader.__iter__().__next__()
                text_support_tar = torch.cat(
                    [clip.tokenize(f'A hyperspectral image of {labels_name_tar[k]}').to(k.device) for k in
                     text_labels]).to('cuda:0')

            # sample datas
            supports, support_labels = support_dataloader.__iter__().__next__()  # (5, 100, 9, 9)
            querys, query_labels = query_dataloader.__iter__().__next__()  # (75,100,9,9)
            _, text_labels = text_dataloder.__iter__().__next__()
            text_support_src = torch.cat(
                [clip.tokenize(f'A hyperspectral image of {labels_name_src[k]}').to(k.device) for k in text_labels]).to('cuda:0')
            # print(support_labels)
            # calculate features
            supports = torch.from_numpy(cube_to_list(supports.numpy()))
            querys = torch.from_numpy(cube_to_list(querys.numpy()))
            target_data = torch.from_numpy(cube_to_list(target_data.numpy()))
            support_features, support_outputs,text_feature_support, text_outputs = feature_encoder(supports.cuda(),text_support_src)  # torch.Size([409, 32, 7, 3, 3])
            _, _, text_feature_support_tar, text_outputs_tar = feature_encoder(supports.cuda(), text_support_tar)

            query_features, query_outputs = feature_encoder(querys.cuda())  # torch.Size([409, 32, 7, 3, 3])
            target_features, target_outputs = feature_encoder(target_data.cuda(),
                                                              domain='target')  # torch.Size([409, 32, 7, 3, 3])

            # Prototype network
            if SHOT_NUM_PER_CLASS > 1:
                support_proto = support_features.reshape(CLASS_NUM, SHOT_NUM_PER_CLASS, -1).mean(dim=1)  # (9, 160)
            else:
                support_proto = support_features

            # 归一化模块
            text_features = text_feature_support / text_feature_support.norm(dim=1, keepdim=True)
            support_proto = support_proto / support_proto.norm(dim=1, keepdim=True)
            query_features = query_features / query_features.norm(dim=1, keepdim=True)
            target_features = target_features / target_features.norm(dim=1, keepdim=True)
            text_feature_support_tar = text_feature_support_tar/text_feature_support_tar.norm(dim=1, keepdim=True)
            '''domain adaptation'''
            # calculate domain adaptation loss
            features = torch.cat([support_proto, query_features, target_features], dim=0)
            outputs = torch.cat((support_outputs, query_outputs, target_outputs), dim=0)
            softmax_output = nn.Softmax(dim=1)(outputs)

            # set label: source 1; target 0
            domain_label = torch.zeros([supports.shape[0] + querys.shape[0] + target_data.shape[0] , 1]).cuda()
            domain_label[:supports.shape[0] + querys.shape[0]] = 1  # torch.Size([225=9*20+9*4, 100, 9, 9])

            randomlayer_out = random_layer.forward([features, softmax_output])  # torch.Size([225, 1024=32*7*3*3])

            domain_logits = domain_classifier(randomlayer_out, episode)
            domain_loss = domain_criterion(domain_logits, domain_label)
            # domain_loss = 0

            '''domain adaptation for text'''
            features_text = torch.cat([text_features, text_feature_support_tar, support_proto, query_features, target_features], dim=0)
            outputs_text = torch.cat((text_outputs, text_outputs_tar, support_outputs, query_outputs, target_outputs), dim=0)
            softmax_output_text = nn.Softmax(dim=1)(outputs_text)

            domain_label_text = torch.zeros([2 * supports.shape[0] + supports.shape[0] + querys.shape[0] + target_data.shape[0], 1]).cuda()
            domain_label_text[:2 * supports.shape[0]] = 1  # torch.Size([225=9*20+9*4, 100, 9, 9])

            randomlayer_out_text = random_layer.forward([features_text, softmax_output_text])

            domain_logits_text = domain_classifier_text(randomlayer_out_text, episode)  # , label_logits
            domain_loss_text = domain_criterion(domain_logits_text, domain_label_text)

            '''fsl_loss'''
            logits = euclidean_metric(query_features, support_proto)
            f_loss = crossEntropy(logits, query_labels.cuda())

            '''text'''
            image_features = support_proto

            logit_text2proto = euclidean_metric(text_features, support_proto)
            loss_clip_sup = crossEntropy(logit_text2proto, text_labels.cuda())

            logits_text = euclidean_metric(query_features, text_features)
            loss_clip_qry = crossEntropy(logits_text, query_labels.cuda())
            loss_clip = loss_clip_sup + loss_clip_qry
            # total_loss = fsl_loss + domain_loss
            loss = f_loss + loss_clip + domain_loss + 1.0 * domain_loss_text # 0.01

            # Update parameters
            feature_encoder.zero_grad()
            domain_classifier.zero_grad()
            domain_classifier_text.zero_grad()
            loss.backward()
            feature_encoder_optim.step()
            domain_classifier_optim.step()
            domain_classifier_optim_text.step()
            total_hit += torch.sum(torch.argmax(logits, dim=1).cpu() == query_labels).item()
            total_num += querys.shape[0]
        # target domain few-shot + domain adaptation
        else:
            '''Few-shot classification for target domain data set'''
            # get few-shot classification samples
            task = utils.Task(target_da_metatrain_data, TEST_CLASS_NUM, SHOT_NUM_PER_CLASS,
                              QUERY_NUM_PER_CLASS)  # 5， 1，15
            support_dataloader = utils.get_HBKC_data_loader(task, num_per_class=SHOT_NUM_PER_CLASS, split="train",
                                                            shuffle=False)
            query_dataloader = utils.get_HBKC_data_loader(task, num_per_class=QUERY_NUM_PER_CLASS, split="test",
                                                          shuffle=True)
            text_dataloader = utils.get_HBKC_data_loader(task, num_per_class=1, split="train", shuffle=False)
            # sample datas
            supports, support_labels = support_dataloader.__iter__().__next__()  # (5, 100, 9, 9)
            querys, query_labels = query_dataloader.__iter__().__next__()  # (75,100,9,9)
            _, text_labels = text_dataloader.__iter__().__next__()
            text_support_tar = torch.cat(
                [clip.tokenize(f'A hyperspectral image of {labels_name_tar[k]}').to(k.device) for k in text_labels]).to('cuda:0')
            # calculate features
            supports = torch.from_numpy(cube_to_list(supports.numpy()))
            querys = torch.from_numpy(cube_to_list(querys.numpy()))
            source_data = torch.from_numpy(cube_to_list(source_data.numpy()))
            support_features, support_outputs , text_feature_support , text_outputs = feature_encoder(supports.cuda(),text_support_tar,
                                                                domain='target')  # torch.Size([409, 32, 7, 3, 3])
            _, _, text_feature_support_src, text_outputs_src = feature_encoder(supports.cuda(), text_support_src, domain='target')
            query_features, query_outputs = feature_encoder(querys.cuda(), None,
                                                            domain='target')  # torch.Size([409, 32, 7, 3, 3])
            source_features, source_outputs = feature_encoder(source_data.cuda())  # torch.Size([409, 32, 7, 3, 3])

            # Prototype network
            if SHOT_NUM_PER_CLASS > 1:
                support_proto = support_features.reshape(CLASS_NUM, SHOT_NUM_PER_CLASS, -1).mean(dim=1)  # (9, 160)
            else:
                support_proto = support_features

            # 归一化模块
            text_features = text_feature_support / text_feature_support.norm(dim=1, keepdim=True)
            support_proto = support_proto / support_proto.norm(dim=1, keepdim=True)
            query_features = query_features / query_features.norm(dim=1, keepdim=True)
            source_features = source_features / source_features.norm(dim=1, keepdim=True)
            text_feature_support_src = text_feature_support_src/text_feature_support_src.norm(dim=1, keepdim=True)

            '''domain adaptation for Image'''
            features = torch.cat([support_proto, query_features, source_features], dim=0)
            outputs = torch.cat((support_outputs, query_outputs, source_outputs), dim=0)
            softmax_output = nn.Softmax(dim=1)(outputs)

            domain_label = torch.zeros([supports.shape[0] + querys.shape[0] + source_features.shape[0], 1]).cuda()
            domain_label[supports.shape[0] + querys.shape[0]:] = 1  # torch.Size([225=9*20+9*4, 100, 9, 9])

            randomlayer_out = random_layer.forward([features, softmax_output])  # torch.Size([225, 1024=32*7*3*3])

            domain_logits = domain_classifier(randomlayer_out, episode)  # , label_logits
            domain_loss = domain_criterion(domain_logits, domain_label)

            '''domain adaptation for text'''
            features_text = torch.cat([text_features, text_feature_support_src, support_proto, query_features, source_features], dim=0)
            outputs_text = torch.cat((text_outputs, text_outputs_src, support_outputs, query_outputs, source_outputs), dim=0)
            softmax_output_text = nn.Softmax(dim=1)(outputs_text)

            domain_label_text = torch.zeros([2*supports.shape[0] + supports.shape[0] + querys.shape[0] + source_features.shape[0], 1]).cuda()
            domain_label_text[:2 * supports.shape[0]] = 1  # torch.Size([225=9*20+9*4, 100, 9, 9])

            randomlayer_out_text = random_layer.forward([features_text, softmax_output_text])

            domain_logits_text = domain_classifier_text(randomlayer_out_text, episode)  # , label_logits
            domain_loss_text = domain_criterion(domain_logits_text, domain_label_text)

            '''fsl_loss'''
            logits = euclidean_metric(query_features, support_proto)
            f_loss = crossEntropy(logits, query_labels.cuda())
            '''text'''
            image_features = support_proto

            logit_text2proto = euclidean_metric(text_features, support_proto)
            loss_clip_sup = crossEntropy(logit_text2proto, text_labels.cuda())
            logits_text = euclidean_metric(query_features, text_features)
            loss_clip_qry = crossEntropy(logits_text, query_labels.cuda())
            loss_clip = loss_clip_sup + loss_clip_qry
            # total_loss = fsl_loss + domain_loss
            loss = f_loss + loss_clip + domain_loss + 1.0 * domain_loss_text # 0.01 0.5=78;0.25=80;0.01=80

            # Update parameters
            feature_encoder.zero_grad()
            domain_classifier.zero_grad()
            domain_classifier_text.zero_grad()
            loss.backward()
            feature_encoder_optim.step()
            domain_classifier_optim.step()
            domain_classifier_optim_text.step()
            total_hit += torch.sum(torch.argmax(logits, dim=1).cpu() == query_labels).item()
            total_num += querys.shape[0]

        if (episode + 1) % 100 == 0:  # display
            train_loss.append(loss.item())
            print('episode {:>3d}:  image domain loss: {:6.4f}, text domain loss: {:6.4f} fsl loss: {:6.4f}, acc {:6.4f}, loss: {:6.4f}'.format(
                episode + 1, \
                domain_loss.item(),
                domain_loss_text.item(),
                f_loss.item(),
                total_hit / total_num,
                loss.item()))

        if (episode + 1) % 1000 == 0 or episode == 0:
            # test
            print("Testing ...")
            train_end = time.time()
            feature_encoder.eval()
            total_rewards = 0
            counter = 0
            accuracies = []
            predict = np.array([], dtype=np.int64)
            labels = np.array([], dtype=np.int64)

            train_datas, train_labels = train_loader.__iter__().__next__()
            train_datas = torch.from_numpy(cube_to_list(train_datas.numpy()))
            train_features, _ = feature_encoder(Variable(train_datas).cuda(), domain='target')  # (45, 160)

            max_value = train_features.max()  # 89.67885
            min_value = train_features.min()  # -57.92479
            print(max_value.item())
            print(min_value.item())
            train_features = (train_features - min_value) * 1.0 / (max_value - min_value)

            KNN_classifier = KNeighborsClassifier(n_neighbors=1)
            KNN_classifier.fit(train_features.cpu().detach().numpy(), train_labels)  # .cpu().detach().numpy()
            for test_datas, test_labels in test_loader:
                batch_size = test_labels.shape[0]

                test_datas = torch.from_numpy(cube_to_list(test_datas.numpy()))
                test_features, _ = feature_encoder(Variable(test_datas).cuda(), domain='target')  # (100, 160)
                test_features = (test_features - min_value) * 1.0 / (max_value - min_value)
                predict_labels = KNN_classifier.predict(test_features.cpu().detach().numpy())
                test_labels = test_labels.numpy()
                rewards = [1 if predict_labels[j] == test_labels[j] else 0 for j in range(batch_size)]

                total_rewards += np.sum(rewards)
                counter += batch_size

                predict = np.append(predict, predict_labels)
                labels = np.append(labels, test_labels)

                accuracy = total_rewards / 1.0 / counter  #
                accuracies.append(accuracy)

            test_accuracy = 100. * total_rewards / len(test_loader.dataset)

            print('\t\tAccuracy: {}/{} ({:.2f}%)\n'.format(total_rewards, len(test_loader.dataset),
                                                           100. * total_rewards / len(test_loader.dataset)))
            test_end = time.time()

            # Training mode
            feature_encoder.train()
            if test_accuracy > last_accuracy:
                # save networks
                torch.save(feature_encoder.state_dict(),
                           str("checkpoints/DFSL_feature_encoder_" + "PU_" + str(run) + "iter_" + str(
                               TEST_LSAMPLE_NUM_PER_CLASS) + "shot.pkl"))
                print("save networks for episode:", episode + 1)
                last_accuracy = test_accuracy
                best_episdoe = episode

                acc[run] = 100. * total_rewards / len(test_loader.dataset)
                OA = acc
                C = metrics.confusion_matrix(labels, predict)
                A[run, :] = np.diag(C) / np.sum(C, 1, dtype=np.float64)

                k[run] = metrics.cohen_kappa_score(labels, predict)

            print('best episode:[{}], best accuracy={}'.format(best_episdoe + 1, last_accuracy))

    if test_accuracy > best_acc_all:
        best_predict_all = predict
        best_G, best_RandPerm, best_Row, best_Column, best_nTrain = G, RandPerm, Row, Column, nTrain
    print('iter:{} best episode:[{}], best accuracy={}'.format(run, best_episdoe + 1, last_accuracy))
    print('***********************************************************************************')

AA = np.mean(A, 1)

AAMean = np.mean(AA, 0)
AAStd = np.std(AA)

AMean = np.mean(A, 0)
AStd = np.std(A, 0)

OAMean = np.mean(acc)
OAStd = np.std(acc)

kMean = np.mean(k)
kStd = np.std(k)
print("train time per DataSet(s): " + "{:.5f}".format(train_end - train_start))
print("test time per DataSet(s): " + "{:.5f}".format(test_end - train_end))
print("average OA: " + "{:.2f}".format(OAMean) + " +- " + "{:.2f}".format(OAStd))
print("average AA: " + "{:.2f}".format(100 * AAMean) + " +- " + "{:.2f}".format(100 * AAStd))
print("average kappa: " + "{:.4f}".format(100 * kMean) + " +- " + "{:.4f}".format(100 * kStd))
print("accuracy for each class: ")
for i in range(CLASS_NUM):
    print("Class " + str(i) + ": " + "{:.2f}".format(100 * AMean[i]) + " +- " + "{:.2f}".format(100 * AStd[i]))

best_iDataset = 0
for i in range(len(acc)):
    print('{}:{}'.format(i, acc[i]))
    if acc[i] > acc[best_iDataset]:
        best_iDataset = i
print('best acc all={}'.format(acc[best_iDataset]))

#################classification map################################

for i in range(len(best_predict_all)):  # predict ndarray <class 'tuple'>: (9729,)
    best_G[best_Row[best_RandPerm[best_nTrain + i]]][best_Column[best_RandPerm[best_nTrain + i]]] = best_predict_all[
                                                                                                        i] + 1

hsi_pic = np.zeros((best_G.shape[0], best_G.shape[1], 3))
for i in range(best_G.shape[0]):
    for j in range(best_G.shape[1]):
        if best_G[i][j] == 0:
            hsi_pic[i, j, :] = [0, 0, 0]
        if best_G[i][j] == 1:
            hsi_pic[i, j, :] = [0, 0, 1]
        if best_G[i][j] == 2:
            hsi_pic[i, j, :] = [0, 1, 0]
        if best_G[i][j] == 3:
            hsi_pic[i, j, :] = [0, 1, 1]
        if best_G[i][j] == 4:
            hsi_pic[i, j, :] = [1, 0, 0]
        if best_G[i][j] == 5:
            hsi_pic[i, j, :] = [1, 0, 1]
        if best_G[i][j] == 6:
            hsi_pic[i, j, :] = [1, 1, 0]
        if best_G[i][j] == 7:
            hsi_pic[i, j, :] = [0.5, 0.5, 1]
        if best_G[i][j] == 8:
            hsi_pic[i, j, :] = [0.65, 0.35, 1]
        if best_G[i][j] == 9:
            hsi_pic[i, j, :] = [0.75, 0.5, 0.75]
        if best_G[i][j] == 10:
            hsi_pic[i, j, :] = [0.75, 1, 0.5]
        if best_G[i][j] == 11:
            hsi_pic[i, j, :] = [0.5, 1, 0.65]
        if best_G[i][j] == 12:
            hsi_pic[i, j, :] = [0.65, 0.65, 0]
        if best_G[i][j] == 13:
            hsi_pic[i, j, :] = [0.75, 1, 0.65]
        if best_G[i][j] == 14:
            hsi_pic[i, j, :] = [0, 0, 0.5]
        if best_G[i][j] == 15:
            hsi_pic[i, j, :] = [0, 1, 0.75]
        if best_G[i][j] == 16:
            hsi_pic[i, j, :] = [0.5, 0.75, 1]

utils.classification_map(hsi_pic[4:-4, 4:-4, :], best_G[4:-4, 4:-4], 24,
                         "classificationMap/HT_{}shot.png".format(TEST_LSAMPLE_NUM_PER_CLASS))
