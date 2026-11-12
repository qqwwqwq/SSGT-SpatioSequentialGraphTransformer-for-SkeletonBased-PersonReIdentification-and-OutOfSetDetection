"""
Skeleton Data Loading and Graph Construction.

Loads the coordinate files produced by preprocessing.py and preserves the
original SSGT/TranSG tuple interface, skeleton ordering, class sampling, and
Laplacian encodings for the final WUT, KGBD, and KS20 protocols.
"""

import numpy as np
import scipy.sparse
from sklearn.preprocessing import label_binarize

import paths
from randomness import resolve_seed

# === Data Loading ===

def data_variant_tag(time_step, data_subdir):
    """Derive a checkpoint suffix from a data directory name.

    Args:
        time_step (int): Window length.
        data_subdir (str): Data directory, for example ``6_10fps``.

    Returns:
        str: Variant suffix, for example ``10fps``, or an empty string.
    """
    sub = str(data_subdir or '')
    if not sub or sub == str(int(time_step)):
        return ''
    prefix = str(int(time_step)) + '_'
    return sub[len(prefix):] if sub.startswith(prefix) else sub


def gen_train_data(
    dataset, split, time_step, nb_nodes, nhood, global_att, batch_size,
    view='', reverse='0', D=False, graph=False, norm=True, L3=True,
    enc_k=10, data_subdir='', sampling_seed=None,
):
    """Load skeleton windows and construct graph representations.

    Args:
        dataset (str): One of WUT, KGBD, or KS20.
        split (str): Probe/gallery protocol name, or ``unknown`` for OOS data.
        time_step (int): Number of frames in each window.
        nb_nodes (int): Number of input joints (WUT: 19, KGBD: 20, KS20: 25).
        nhood (int): Neighborhood depth for graph attention masks.
        global_att (bool): Use fully connected attention masks.
        batch_size (int): Batch size used by the inherited class sampler.
        view (str): Optional ``view_<view>`` subdirectory.
        reverse (str): ``1`` selects target coordinates; ``0`` selects source.
        D (bool): Legacy descriptor option; this release requires False.
        graph (bool): Legacy graph option; this release requires False.
        norm (bool): Center coordinates on input joint zero.
        L3 (bool): Return joint, part, and body representations.
        enc_k (int): Number of joint Laplacian positional eigenvectors.
        data_subdir (str): Directory below the dataset, defaulting to time_step.
        sampling_seed (int or None): Seed for balanced sampling and training
            permutation. None or -1 selects fresh entropy. Pass the same
            resolved seed to every split loaded in one run for alignment.

    Returns:
        tuple: With the SSGT defaults (D=False, graph=False, L3=True), 26
        entries in the original order. Joint windows at indices 0 and 6
        have shape (N, T, J, 3), labels at 5 and 11 are one-hot, joint
        adjacency is at 12, joint positional encoding is at 14, and the
        configured number of classes is the final entry. Unknown identities
        have all-zero label rows.

    Raises:
        NotImplementedError: If a legacy D or graph branch is requested.
        ValueError: If the dataset, identity count, or sampling setup is invalid.

    Notes:
        Files are resolved under ``paths.data_root()``. KGBD joints retain
        the original permutation, applied after root centering. Non-KGBD
        training sets retain the original balanced sampling behavior.
        Only load preprocessing files from a trusted source: the identity
        dictionaries use NumPy object serialization.
    """
    if dataset not in {'WUT', 'KGBD', 'KS20'}:
        raise ValueError('This release supports WUT, KGBD, and KS20 preprocessed data.')
    if D or graph:
        raise NotImplementedError(
            'The published SSGT loader supports D=False and graph=False. '
            'Legacy descriptor and alternate graph branches are not implemented.'
        )
    if time_step <= 0 or batch_size <= 0:
        raise ValueError('time_step and batch_size must be positive.')

    def get_data(dimension, fr):
        if reverse == '1':
            used_data = 'target'
        else:
            used_data = 'source'
        input_path = dataset_dir / 'train_npy_data' / f'{used_data}_{dimension}_{dataset}_{fr}.npy'
        input_data = np.load(input_path, allow_pickle=True)
        input_data = input_data.reshape([-1, time_step, nb_nodes])
        if norm:
            spine_pos = input_data[:, :, 0]
            spine_pos = np.expand_dims(spine_pos, -1)
            input_data = input_data - spine_pos
        if split == 'unknown':
            t_used_data = 'source'
            t_base_dir = 'unknown_npy_data'
            t_path = dataset_dir / t_base_dir / f't_{t_used_data}_{dimension}_{dataset}_{fr}.npy'
        else:
            t_used_data = used_data
            t_base_dir = 'test_npy_data/' + split
            t_path = dataset_dir / t_base_dir / f't_{t_used_data}_{dimension}_{dataset}_{fr}.npy'
        t_input_data = np.load(t_path, allow_pickle=True)
        t_input_data = t_input_data.reshape([-1, time_step, nb_nodes])
        if norm is True:
            t_spine_pos = t_input_data[:, :, 0]
            t_spine_pos = np.expand_dims(t_spine_pos, -1)
            t_input_data = t_input_data - t_spine_pos
        return (input_data, t_input_data)

    def normalized_laplacian(adj_matrix):
        R = np.sum(adj_matrix, axis=1)
        R_sqrt = 1 / np.sqrt(R)
        D_sqrt = np.diag(R_sqrt)
        I = np.eye(adj_matrix.shape[0])
        return I - np.matmul(np.matmul(D_sqrt, adj_matrix), D_sqrt)
    sub_dir = str(data_subdir) if data_subdir else str(time_step)
    dataset_dir = paths.data_root() / dataset / sub_dir
    if view != '':
        dataset_dir /= f'view_{view}'
    (input_data_x, t_input_data_x) = get_data('x', fr=time_step)
    (input_data_y, t_input_data_y) = get_data('y', fr=time_step)
    (input_data_z, t_input_data_z) = get_data('z', fr=time_step)
    X_train = np.concatenate([input_data_x, input_data_y, input_data_z], axis=-1)
    X_test = np.concatenate([t_input_data_x, t_input_data_y, t_input_data_z], axis=-1)
    train_dir = dataset_dir / 'train_npy_data'
    ids = np.load(train_dir / f'ids_{dataset}_{time_step}.npy', allow_pickle=True).item()
    if len(ids) < 3:
        raise ValueError(
            'The inherited multiclass encoding requires at least three known '
            'training identities. Binary/custom protocols need an explicit encoding adaptation.'
        )
    if split == 'unknown':
        test_dir = dataset_dir / 'unknown_npy_data'
    else:
        test_dir = dataset_dir / 'test_npy_data' / split
    t_fid_path = test_dir / f'frame_id_{dataset}_{time_step}.npy'
    y_train = np.load(train_dir / f'frame_id_{dataset}_{time_step}.npy', allow_pickle=True)
    y_test = np.load(t_fid_path, allow_pickle=True)
    sampling_rng = np.random.RandomState(resolve_seed(sampling_seed))
    if dataset != 'KGBD':
        (X_train, y_train) = class_samp_gen(X_train, y_train, ids, batch_size, rng=sampling_rng)
    rand_p = sampling_rng.permutation(X_train.shape[0])
    X_train = X_train[rand_p]
    y_train = y_train[rand_p]
    ids_keys = sorted(list(ids.keys()))
    classes = [i for i in ids_keys]
    y_train = label_binarize(y_train, classes=classes)
    y_test = label_binarize(y_test, classes=classes)
    # Coordinate files are axis-major; the model expects (N, T, J, 3).
    X_train_J = X_train.reshape([-1, time_step, 3, nb_nodes])
    X_train_J = np.transpose(X_train_J, [0, 1, 3, 2])
    X_test_J = X_test.reshape([-1, time_step, 3, nb_nodes])
    X_test_J = np.transpose(X_test_J, [0, 1, 3, 2])
    # Retain the original Kinect-to-model joint ordering.
    if dataset == 'KGBD':
        X_train_J = X_train_J[:, :, [11, 10, 1, 0, 2, 4, 6, 8, 3, 5, 7, 9, 12, 14, 16, 18, 13, 15, 17, 19], :]
        X_test_J = X_test_J[:, :, [11, 10, 1, 0, 2, 4, 6, 8, 3, 5, 7, 9, 12, 14, 16, 18, 13, 15, 17, 19], :]
    if L3:
        X_train_P = reduce2part(X_train_J, nb_nodes)
        X_train_B = reduce2body(X_train_J, nb_nodes)
        X_test_P = reduce2part(X_test_J, nb_nodes)
        X_test_B = reduce2body(X_test_J, nb_nodes)

    if dataset == 'KS20':
        j_pair_1 = np.array([3, 2, 20, 8, 8, 9, 10, 9, 11, 10, 4, 20, 4, 5, 5, 6, 6, 7, 1, 20, 1, 0,
            16, 0, 12, 0, 16, 17, 12, 13, 17, 18, 19, 18, 13, 14, 14, 15, 2, 20, 11, 23, 10, 24, 7,
            21, 6, 22])
        j_pair_2 = np.array([2, 3, 8, 20, 9, 8, 9, 10, 10, 11, 20, 4, 5, 4, 6, 5, 7, 6, 20, 1, 0, 1,
            0, 16, 0, 12, 17, 16, 13, 12, 18, 17, 18, 19, 14, 13, 15, 14, 20, 2, 23, 11, 24, 10,
            21, 7, 22, 6])
        con_matrix = np.ones([48])
        adj_joint = scipy.sparse.coo_matrix((con_matrix, (j_pair_1, j_pair_2)), shape=(nb_nodes, nb_nodes)).toarray()
    elif dataset == 'WUT':
        j_pair_1 = np.array([0, 1, 1, 2, 2, 3, 2, 4, 4, 5, 5, 6, 6, 7, 2, 8, 8, 9, 9, 10, 10, 11, 0,
            12, 12, 13, 13, 14, 0, 15, 15, 16, 16, 17, 3, 18])
        j_pair_2 = np.array([1, 0, 2, 1, 3, 2, 4, 2, 5, 4, 6, 5, 7, 6, 8, 2, 9, 8, 10, 9, 11, 10,
            12, 0, 13, 12, 14, 13, 15, 0, 16, 15, 17, 16, 18, 3])
        con_matrix = np.ones([36])
        adj_joint = scipy.sparse.coo_matrix((con_matrix, (j_pair_1, j_pair_2)), shape=(nb_nodes, nb_nodes)).toarray()
    else:
        j_pair_1 = np.array([3, 2, 2, 8, 8, 9, 10, 9, 11, 10, 4, 2, 4, 5, 5, 6, 6, 7, 1, 2, 1, 0,
            16, 0, 12, 0, 16, 17, 12, 13, 17, 18, 19, 18, 13, 14, 14, 15])
        j_pair_2 = np.array([2, 3, 8, 2, 9, 8, 9, 10, 10, 11, 2, 4, 5, 4, 6, 5, 7, 6, 2, 1, 0, 1, 0,
            16, 0, 12, 17, 16, 13, 12, 18, 17, 18, 19, 14, 13, 15, 14])
        con_matrix = np.ones([38])
        adj_joint = scipy.sparse.coo_matrix((con_matrix, (j_pair_1, j_pair_2)), shape=(nb_nodes, nb_nodes)).toarray()
    k = enc_k
    L = normalized_laplacian(adj_joint)
    (EigVal, EigVec) = np.linalg.eig(L)
    idx = EigVal.argsort()
    (EigVal, EigVec) = (EigVal[idx], np.real(EigVec[:, idx]))
    pos_enc_ori = EigVec[:, 1:k + 1]
    if L3:
        p_pair_1 = np.array([5, 6, 5, 8, 6, 7, 8, 9, 5, 4, 4, 2, 4, 0, 2, 3, 1, 0])
        p_pair_2 = np.array([6, 5, 8, 5, 7, 6, 9, 8, 4, 5, 2, 4, 0, 4, 3, 2, 0, 1])
        con_matrix = np.ones([18])
        adj_part = scipy.sparse.coo_matrix((con_matrix, (p_pair_1, p_pair_2)), shape=(10, 10)).toarray()
        b_pair_1 = np.array([2, 3, 2, 4, 2, 1, 2, 0])
        b_pair_2 = np.array([3, 2, 4, 2, 1, 2, 0, 2])
        con_matrix = np.ones([8])
        adj_body = scipy.sparse.coo_matrix((con_matrix, (b_pair_1, b_pair_2)), shape=(5, 5)).toarray()
        k = 9
        L = normalized_laplacian(adj_part)
        (EigVal, EigVec) = np.linalg.eig(L)
        idx = EigVal.argsort()
        (EigVal, EigVec) = (EigVal[idx], np.real(EigVec[:, idx]))
        pos_enc_part = EigVec[:, 1:k + 1]
        k = 4
        L = normalized_laplacian(adj_body)
        (EigVal, EigVec) = np.linalg.eig(L)
        idx = EigVal.argsort()
        (EigVal, EigVec) = (EigVal[idx], np.real(EigVec[:, idx]))
        pos_enc_body = EigVec[:, 1:k + 1]
    if global_att:
        if dataset == 'KS20':
            node_num = 25
        elif dataset == 'WUT':
            node_num = 19
        else:
            node_num = 20
        adj_joint = np.ones([node_num, node_num])
        adj_part = np.ones([10, 10])
        adj_body = np.ones([5, 5])
    nb_classes = {'WUT': 25, 'KGBD': 164, 'KS20': 20}[dataset]
    adj_joint = adj_joint[np.newaxis]
    if not L3:
        biases_joint = adj_to_bias(adj_joint, [nb_nodes], nhood=nhood)
        return (X_train_J, 0, 0, 0, 0, y_train, X_test_J, 0, 0, 0, 0, y_test,
            adj_joint, biases_joint, 0, 0, 0, 0, 0, 0, 0, 0, nb_classes)
    adj_part = adj_part[np.newaxis]
    adj_body = adj_body[np.newaxis]
    return (X_train_J, X_train_P, X_train_B, 0, 0, y_train, X_test_J, X_test_P,
        X_test_B, 0, 0, y_test, adj_joint, 0, pos_enc_ori, adj_part, 0,
        pos_enc_part, adj_body, 0, pos_enc_body, 0, 0, 0, 0, nb_classes)


# === Skeleton Graph Representations ===


def reduce2part(X, joint_num=20):
    """Average joint coordinates into 10 body components.

    Args:
        X (np.ndarray): Joint windows of shape (N, T, J, 3).
        joint_num (int): Number of joints in the dataset skeleton.

    Returns:
        np.ndarray: Component coordinates of shape (N, T, 10, 3).
    """
    if joint_num == 25:
        left_leg_up = [12, 13]
        left_leg_down = [14, 15]
        right_leg_up = [16, 17]
        right_leg_down = [18, 19]
        torso = [0, 1]
        head = [2, 3, 20]
        left_arm_up = [4, 5]
        left_arm_down = [6, 7, 21, 22]
        right_arm_up = [8, 9]
        right_arm_down = [10, 11, 23, 24]
    elif joint_num == 20:
        left_leg_up = [12, 13]
        left_leg_down = [14, 15]
        right_leg_up = [16, 17]
        right_leg_down = [18, 19]
        torso = [0, 1]
        head = [2, 3]
        left_arm_up = [4, 5]
        left_arm_down = [6, 7]
        right_arm_up = [8, 9]
        right_arm_down = [10, 11]
    elif joint_num == 19:
        left_leg_up = [12, 13]
        left_leg_down = [14]
        right_leg_up = [15, 16]
        right_leg_down = [17]
        torso = [0, 1, 2]
        head = [3, 18]
        left_arm_up = [4, 5]
        left_arm_down = [6, 7]
        right_arm_up = [8, 9]
        right_arm_down = [10, 11]
    x_torso = np.mean(X[:, :, torso, :], axis=2)
    x_leftlegup = np.mean(X[:, :, left_leg_up, :], axis=2)
    x_leftlegdown = np.mean(X[:, :, left_leg_down, :], axis=2)
    x_rightlegup = np.mean(X[:, :, right_leg_up, :], axis=2)
    x_rightlegdown = np.mean(X[:, :, right_leg_down, :], axis=2)
    x_head = np.mean(X[:, :, head, :], axis=2)
    x_leftarmup = np.mean(X[:, :, left_arm_up, :], axis=2)
    x_leftarmdown = np.mean(X[:, :, left_arm_down, :], axis=2)
    x_rightarmup = np.mean(X[:, :, right_arm_up, :], axis=2)
    x_rightarmdown = np.mean(X[:, :, right_arm_down, :], axis=2)
    X_part = np.concatenate((x_leftlegup, x_leftlegdown, x_rightlegup, x_rightlegdown, x_torso, x_head, x_leftarmup, x_leftarmdown, x_rightarmup, x_rightarmdown), axis=-1).reshape([X.shape[0], X.shape[1], 10, 3])
    return X_part


def reduce2body(X, joint_num=20):
    """Average joint coordinates into 5 body components.

    Args:
        X (np.ndarray): Joint windows of shape (N, T, J, 3).
        joint_num (int): Number of joints in the dataset skeleton.

    Returns:
        np.ndarray: Component coordinates of shape (N, T, 5, 3).
    """
    if joint_num == 25:
        left_leg = [12, 13, 14, 15]
        right_leg = [16, 17, 18, 19]
        torso = [0, 1, 2, 3, 20]
        left_arm = [4, 5, 6, 7, 21, 22]
        right_arm = [8, 9, 10, 11, 23, 24]
    elif joint_num == 20:
        left_leg = [12, 13, 14, 15]
        right_leg = [16, 17, 18, 19]
        torso = [0, 1, 2, 3]
        left_arm = [4, 5, 6, 7]
        right_arm = [8, 9, 10, 11]
    elif joint_num == 19:
        left_leg = [12, 13, 14]
        right_leg = [15, 16, 17]
        torso = [0, 1, 2, 3, 18]
        left_arm = [4, 5, 6, 7]
        right_arm = [8, 9, 10, 11]
    x_torso = np.mean(X[:, :, torso, :], axis=2)
    x_leftleg = np.mean(X[:, :, left_leg, :], axis=2)
    x_rightleg = np.mean(X[:, :, right_leg, :], axis=2)
    x_leftarm = np.mean(X[:, :, left_arm, :], axis=2)
    x_rightarm = np.mean(X[:, :, right_arm, :], axis=2)
    X_body = np.concatenate((x_leftleg, x_rightleg, x_torso, x_leftarm, x_rightarm), axis=-1).reshape([X.shape[0], X.shape[1], 5, 3])
    return X_body


# === Sampling and Graph Utilities ===


def class_samp_gen(X, y, ids_, batch_size, rng=None):
    """Construct the inherited balanced training sample sequence.

    Args:
        X (np.ndarray): Training windows.
        y (np.ndarray): Per-window identity labels.
        ids_ (dict): Identity-to-window-index mapping.
        batch_size (int): Requested batch size; must cover all classes.
        rng (numpy.random.RandomState or None): Isolated sampling generator.
            None creates a generator from fresh entropy.

    Returns:
        tuple: Sampled windows and labels. Sampling follows the original
        implementation, including its number of batches and replacement.

    Raises:
        ValueError: If the batch cannot cover the known classes, training
        contains fewer windows than one batch, or an identity has no windows.
    """
    class_num = len(ids_.keys())
    if class_num == 0 or batch_size < class_num:
        raise ValueError(
            f'Balanced sampling requires batch_size >= known classes; '
            f'got batch_size={batch_size}, classes={class_num}.'
        )
    if y.shape[0] < batch_size:
        raise ValueError(
            f'Balanced sampling requires at least one batch of training windows; '
            f'got {y.shape[0]} windows and batch_size={batch_size}.'
        )
    if any(len(indices) == 0 for indices in ids_.values()):
        raise ValueError('Each known training identity must have at least one window.')
    if rng is None:
        rng = np.random.RandomState(resolve_seed())
    ids_ = sorted(ids_.items(), key=lambda item: item[0])
    all_batch_X = []
    all_batch_y = []
    total = y.shape[0]
    batch_num = total // batch_size * 2
    class_in_bacth = class_num
    batch_per_class = batch_size // class_in_bacth
    class_cnt = class_in_bacth
    for i in range(batch_num):
        batch_X = []
        batch_y = []
        for (k, v) in ids_[class_cnt - class_in_bacth:class_cnt]:
            if len(v[batch_per_class * i:batch_per_class * (i + 1)]) < batch_per_class:
                rand_ind = rng.choice(len(v), batch_per_class)
                v_array = np.array(v)
                samp_per_class = v_array[rand_ind].tolist()
                batch_X.extend(samp_per_class)
            else:
                batch_X.extend(v[batch_per_class * i:batch_per_class * (i + 1)])
            batch_y.extend(batch_per_class * [k])
        if class_cnt + class_in_bacth > class_num and class_cnt <= class_num:
            class_cnt = class_num
        else:
            class_cnt = class_cnt + class_in_bacth
        all_batch_X.extend(batch_X)
        all_batch_y.extend(batch_y)
    X_train = X[all_batch_X]
    y_train = np.array(all_batch_y)
    return (X_train, y_train)


def adj_to_bias(adj, sizes, nhood=1):
    """Convert adjacency matrices into additive attention masks.

    Args:
        adj (np.ndarray): Adjacency arrays of shape (G, J, J).
        sizes (list): Number of valid nodes per graph.
        nhood (int): Maximum neighborhood depth, including self-loops.

    Returns:
        np.ndarray: Zero for reachable pairs and -1e9 for masked pairs.
    """
    nb_graphs = adj.shape[0]
    mt = np.empty(adj.shape)
    for g in range(nb_graphs):
        mt[g] = np.eye(adj.shape[1])
        for _ in range(nhood):
            mt[g] = np.matmul(mt[g], adj[g] + np.eye(adj.shape[1]))
        for i in range(sizes[g]):
            for j in range(sizes[g]):
                if mt[g][i][j] > 0.0:
                    mt[g][i][j] = 1.0
    return -1000000000.0 * (1.0 - mt)
