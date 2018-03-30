# -*- coding: utf-8 -*-


import numpy as np
import scipy.io
from sklearn.metrics import average_precision_score


# Configure hyper-parameters and datasets
####################################################################
save_name = 'baseline_new_Market1501'
load_path = os.path.join('results', 'features_baseline_new_Market1501.mat')
save_path = os.path.join('results', 'res_'+save_name)
k_neighbor = 1
k_top = 1

# get test data
test_data = scipy.io.loadmat(load_path)
gallery_feature = test_data['gallery_feature']
gallery_label = test_data['gallery_label'].flatten()
gallery_cam = test_data['gallery_cam'].flatten()
query_feature = test_data['query_feature']
query_label = test_data['query_label'].flatten()
query_cam = test_data['query_cam'].flatten()


# Calculate similarity metrics
####################################################################
def get_dismat_euclidean(x, y):
    # calculate the distance matrix among images in x and y
    # x: matrix, ndarray with shape (n,d)
    # y: matrix, ndarray with shape (N,d)
    # dismat: ndarray with shape (n,N)
    x_v = np.sum(x**2, axis=1)
    y_v = np.sum(y**2, axis=1)
    dismat = x_v.reshape((x_v.shape[0], 1)).dot(np.ones((1, y.shape[0]))) + \
             np.ones((x.shape[0], 1)).dot(y_v.reshape((y_v.shape[0], 1)).T) - \
             2*x.dot(y.T)
    dismat = np.sqrt(dismat)
    return dismat

# get distance matrix
qg_mat = get_dismat_euclidean(query_feature, gallery_feature)


# Evaluate
####################################################################
def get_filter_index(query_label, query_cam, gallery_label, gallery_cam):
    # clean wrong samples
    # query_label, query_cam: label, cam of the current image
    # gallery_label, gallery_cam: label, cam of all images in gallery
    # filter_index: indexes that should be kept in score
    query_index = np.argwhere(gallery_label == query_label)
    camera_index = np.argwhere(gallery_cam == query_cam)
    junk_index1, junk_index2 = np.argwhere(gallery_label == -1), np.intersect1d(query_index, camera_index)
    junk_index = np.append(junk_index2, junk_index1)
    # filter the original index by removing junk index
    filter_index = np.in1d(np.arange(gallery_label.shape[0]), junk_index, invert=True)  # get the indexes of junk images
    return filter_index


def get_binary_label(label, lst_label):
    # label: the class label of current queries
    # lst_label: ndarray with shape (m,), the class label of all samples in the gallery
    i = np.argwhere(lst_label == label).flatten()  # i with shape (m,)
    binary_label = np.zeros(lst_label.shape[0])  # true with shape (m,)
    binary_label[i] = 1  # true with shape (m,)
    return binary_label


# get prediction (0 or 1, 1 means true), probability and cmc for each query
lst_cmc = []
query_class = np.unique(query_label)
prob_mat = np.zeros((query_label.shape[0], len(query_class)))  # prob_mat with shape (num_query, class)
for i in range(query_label.shape[0]):
    # filter the index and score
    filter_index = get_filter_index(query_label[i], query_cam[i], gallery_label, gallery_cam)
    filter_score, filter_gallery_label = qg_mat[i, filter_index], gallery_label[filter_index]
    sort_index = np.argsort(filter_score)
    # get prob and cmc, where set(sort_label) = 1 if sort_label == current label else 0
    sort_label = filter_gallery_label[sort_index]
    sort_label = get_binary_label(query_label[i], sort_label)
    neighbors_label_cmc = sort_label[:k_top]
    neighbors_label = sort_label[:k_neighbor]
    # calulate current cmc
    if 1 in neighbors_label_cmc:
        lst_cmc.append(1)
    else:
        lst_cmc.append(0)
    # calculate current ap
    if 1 in neighbors_label:
        prob_mat[i, np.argwhere(query_class == query_label[i])] = np.sum(neighbors_label) / k_top  # get probability of the current class

# calculate Average Precision (AP) and Mean Average Precision (mAP)
lst_ap = []
for i_class, class_now in enumerate(query_class):
    # get binary label and pre
    true_now = get_binary_label(class_now, query_label)
    pre_now = prob_mat[:, i_class]
    ap_now = average_precision_score(true_now, pre_now)
    lst_ap.append(ap_now)
    print('In {}/{}, the AP of class {} is:{}'.format(i_class, len(set(query_label)), class_now, ap_now))
mAP = np.mean(lst_ap)
print('mAP is:{} when the number of neighbors is:{}.'.format(mAP, k_neighbor))

# calculate CMC
CMC = np.mean(lst_cmc)
print('CMC is:{} when top k is set to:{}.'.format(CMC, k_top))

# save the generated data
results = {'mAP': mAP,
           'CMC': CMC,
           'lst_ap': lst_ap,
           'lst_cmc': lst_cmc,
           'k_top': k_top,
           'k_neighbor': k_neighbor}
scipy.io.savemat(save_path, results)
print('Successfully save the results.')
