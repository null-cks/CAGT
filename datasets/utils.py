import numpy as np
from sklearn.model_selection import KFold
import random
from sklearn.model_selection import StratifiedShuffleSplit,StratifiedKFold


# 5K 6:2:2
def train_val_test_split(kfold = 5, fold = 0, n_sub = 1035):

    id = list(range(n_sub))

    random.seed(123)
    random.shuffle(id)

    kf = KFold(n_splits=kfold, random_state=123,shuffle = True)
    kf2 = KFold(n_splits=kfold-1, shuffle=True, random_state = 666)

    test_index = list()
    train_index = list()
    val_index = list()

    for tr,te in kf.split(np.array(id)):
        test_index.append(te)
        tr_id, val_id = list(kf2.split(tr))[0]
        train_index.append(tr[tr_id])
        val_index.append(tr[val_id])

    train_id = train_index[fold]
    test_id = test_index[fold]
    val_id = val_index[fold]

    return train_id.astype(np.int64), val_id.astype(np.int64), test_id.astype(np.int64)

# 5K 8:2
def train_test_splitKFold(kfold=5, random_state=42,n_sub = 1035):
    id = list(range(n_sub))
    random.shuffle(id)
    kf = KFold(n_splits=kfold, random_state=random_state, shuffle=True)

    test_index = list()
    train_index = list()

    for tr, te in kf.split(np.array(id)):
        train_index.append(tr.astype(np.int64))
        test_index.append(te.astype(np.int64))


    return train_index,  test_index


# 5k 7:1:2
def train_val_test_split712(kfold=5, fold=0, n_sub = 10):
    id = list(range(n_sub))

    random.seed(123)
    random.shuffle(id)

    kf = KFold(n_splits=kfold, random_state=0, shuffle=True)
    kf2 = KFold(n_splits=8, random_state=0, shuffle=True)
    # kf3 = KFold(n_splits=kfold-2, random_state=0, shuffle=True)

    test_index = list()
    val_index = list()
    train_index = list()
    index = 0

    for tr1, te1 in kf.split(np.array(id)):
        tr2, te2 = list(kf2.split(tr1))[index]
        # index += 1
        # tr_id, val_id = list(kf3.split(tr2))[0]
        train_index.append(tr1[tr2])
        val_index.append(tr1[te2])
        test_index.append(te1)

    train_id = train_index[fold]
    val_id = val_index[fold]
    test_id = test_index[fold]

    return train_id.astype(np.int64),  val_id.astype(np.int64), test_id.astype(np.int64)

#按站点划分tr_te
def StratifiedShuffleSplit_tr_te(n_splits=5, train_set=0.7, te_set=0.3, random_state=42, n_sub=1009):
    if n_sub == 1009:
        data = np.load('D:/dataset/ABIDE_I/ABIDE_pcp/cpac/filt_noglobal/cc200/NPY/1009_abide.npy',
                       allow_pickle=True).item()
    else:
        data = np.load('D:/dataset/ABIDE_I/ABIDE_pcp/cpac/filt_noglobal/cc200/NPY/1035_abide.npy',
                       allow_pickle=True).item()
    timeseries = data["corr"]
    site = data['site']
    label = data['label']

    train_index = list()
    test_index = list()

    split = StratifiedShuffleSplit(n_splits=n_splits, test_size=te_set, train_size=train_set,
                                   random_state=random_state)
    for tr_id, te_id in split.split(timeseries, site):
        train_index.append(tr_id)
        test_index.append(te_id)

    mean = np.mean(label)
    tr_lab = label[train_index[0]]
    te_lab = label[test_index[0]]
    mean1 = np.mean(tr_lab)
    mean3 = np.mean(te_lab)


    return train_index, test_index


# 按站点分层
def StratifiedShuffleSplit_tr_vl_te(n_splits=5, train_set=0.7, val_set=0.1 ,random_state=0, n_sub=1035):
    if n_sub == 1009:
        data = np.load('D:/dataset/ABIDE_I/ABIDE_pcp/cpac/filt_noglobal/cc200/NPY/1009_abide.npy', allow_pickle=True).item()
    else:
        data = np.load('D:/dataset/ABIDE_I/ABIDE_pcp/cpac/filt_noglobal/cc200/NPY/1035_abide.npy', allow_pickle=True).item()
    timeseries = data["corr"]
    site = data['site']

    train_length = int(n_sub*train_set)
    val_length = int(n_sub*val_set)
    test_length = n_sub-train_length-val_length

    train_index = list()
    timeseries_val_te_list = list()
    site_val_te_list = list()
    valid_index = list()
    test_index = list()
    val_te_index_list = list()


    split = StratifiedShuffleSplit(n_splits=n_splits, test_size=val_length + test_length, train_size=train_length, random_state=random_state)
    for tr_id, te_val_id in split.split(timeseries, site):
        train_index.append(tr_id)
        val_te_index_list.append(te_val_id)
        timeseries_val_te_list.append(timeseries[te_val_id])
        site_val_te_list.append(site[te_val_id])

    for timeseries_val_te, site_val_te, val_te_index in zip(timeseries_val_te_list, site_val_te_list, val_te_index_list):
        split = StratifiedShuffleSplit(n_splits=1, test_size=val_length, train_size=test_length,random_state=random_state)
        for te_id, val_id in split.split(timeseries_val_te, site_val_te):
            test_index.append(val_te_index[te_id])
            valid_index.append(val_te_index[val_id])

    return train_index, valid_index, test_index

#按站点划分k折
def StratifiedKFold_tr_te(n_splits=5, random_state=42, n_sub=1035):
    if n_sub == 1009:
        data = np.load('D:/dataset/ABIDE_I/ABIDE_pcp/cpac/filt_noglobal/cc200/NPY/1009_abide.npy', allow_pickle=True).item()
    else:
        data = np.load('D:/dataset/ABIDE_I/ABIDE_pcp/cpac/filt_noglobal/cc200/NPY/1035_abide.npy', allow_pickle=True).item()
    X = data["corr"]
    site = data['site']
    train_index = list()
    test_index = list()

    skf = StratifiedKFold(n_splits=n_splits, random_state=random_state, shuffle=True)
    for i, (tr_id, te_id) in enumerate(skf.split(X, site)):
        train_index.append(tr_id.astype(np.int64))
        test_index.append(te_id.astype(np.int64))


    return train_index, test_index

def StratifiedKFold_tr_te_lab(n_splits=5, random_state=42, n_sub=1035, x=None, label=None):

    train_index = list()
    test_index = list()
    x = x[:n_sub,:]
    skf = StratifiedKFold(n_splits=n_splits, random_state=random_state, shuffle=True)
    for i, (tr_id, te_id) in enumerate(skf.split(x, label)):
        train_index.append(tr_id.astype(np.int64))
        test_index.append(te_id.astype(np.int64))

    return train_index, test_index

