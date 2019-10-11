import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score , average_precision_score
import tensorflow as tf



def get_acc(y_true, y_pred):
    y_true = y_true.flatten()
    y_pred = y_pred.flatten()
    y_true = np.where(y_true >= 0.001, 1, 0)
    y_pred = np.where(y_pred >= 0.5, 1, 0)
    return accuracy_score(y_true , y_pred)

def get_precision(y_true, y_pred):
    y_true = y_true.flatten()
    y_pred = y_pred.flatten()
    y_true = np.where(y_true >= 0.001, 1, 0)
    y_pred = np.where(y_pred >= 0.5, 1, 0)
    return precision_score(y_true , y_pred)

def get_recall(y_true, y_pred):
    y_true = y_true.flatten()
    y_pred = y_pred.flatten()
    y_true = np.where(y_true >= 0.001, 1, 0)
    y_pred = np.where(y_pred >= 0.5, 1, 0)
    return recall_score(y_true , y_pred)

def get_auroc(y_true, y_pred):
    try:
        y = y_true.flatten()
        y_true = np.where(y >= 0.001, 1, 0)
        y_pred = y_pred.flatten()
        return roc_auc_score(y_true , y_pred)
    except:
        print('auc error!')
        print('y_true:' , np.max(y_true) , np.min(y_true) , np.average(y_true))
        print('y_pred:' , np.max(y_pred) , np.min(y_pred) , np.average(y_pred))
        return 100

def get_aupr(y_true, y_pred):
    try:
        y_true = y_true.flatten()
        y_pred = y_pred.flatten()
        y_true = np.where(y_true >= 0.001, 1, 0)
        return average_precision_score(y_true , y_pred)
    except:
        print('aupr error!')
        print('y_true:' , np.max(y_true) , np.min(y_true) , np.average(y_true))
        print('y_pred:' , np.max(y_pred) , np.min(y_pred) , np.average(y_pred))
        return 100


def get_cindex(y_true, y_pred):
    g = tf.subtract(tf.expand_dims(y_pred, -1), y_pred)
    g = tf.cast(g == 0.0, tf.float32) * 0.5 + tf.cast(g > 0.0, tf.float32)

    f = tf.subtract(tf.expand_dims(y_true, -1), y_true) > 0.0
    f = tf.matrix_band_part(tf.cast(f, tf.float32), -1, 0)

    g = tf.reduce_sum(tf.multiply(g, f))
    f = tf.reduce_sum(f)

    return tf.where(tf.equal(g, 0), 0.0, g/f) #select

def _to_tensor(x, dtype):
    """Convert the input `x` to a tensor of type `dtype`.

    # Arguments
        x: An object to be converted (numpy array, list, tensors).
        dtype: The destination type.

    # Returns
        A tensor.
    """
    return tf.convert_to_tensor(x, dtype=dtype)


def get_nce(y_true, y_pred, from_logits=False, axis=-1):
    output_dimensions = list(range(len(y_pred.get_shape())))
    if axis != -1 and axis not in output_dimensions:
        raise ValueError(
            '{}{}{}'.format(
                'Unexpected channels axis {}. '.format(axis),
                'Expected to be -1 or one of the axes of `output`, ',
                'which has {} dimensions.'.format(len(y_pred.get_shape()))))
    y_pred /= tf.reduce_sum(y_pred, axis, True)
    _epsilon = _to_tensor(1e-7, y_pred.dtype.base_dtype)
    output = tf.clip_by_value(y_pred, _epsilon, 1. - _epsilon)
    return - tf.reduce_sum(y_true * tf.log(output), axis)

def dcg_score(y_true, y_score, k=5):
    order = np.argsort(y_score)[::-1]
    y_true = np.take(y_true, order[:k])
    gain = 2 ** y_true - 1
    discounts = np.log2(np.arange(len(y_true)) + 2)

    return np.sum(gain / discounts)


def get_ndcg(y_true, y_pred, k=5):
    y_true = y_true.flatten()
    y_pred = y_pred.flatten()
    y_true = np.where(y_true >= 0.001, 1, 0)
    if len(y_true) == len(y_pred):
        length = len(y_true)
    else:
        print("y_true and y_score have different value ranges")
        return

    matrix_true = np.zeros([length , 3])
    matrix_pred = np.zeros([length , 3])

    for i in range(length):
        if y_true[i] == 0:
            matrix_true[i][0] = 1
        else:
            matrix_true[i][1] = 1
        matrix_pred[i][0] = y_pred[i]
        matrix_pred[i][1] = 1 - y_pred[i]
    scores = []
    for y_value_true, y_value_score in zip(matrix_true, matrix_pred):
        actual = dcg_score(y_value_true, y_value_score, k)
        best = dcg_score(y_value_true, y_value_true, k)
        # print(best)
        scores.append(actual / best)
    return np.mean(scores)
'''
x = np.random.rand(18000)
#x[3] = 2
y = np.random.rand(18000)
#y = y/np.sum(y)
x = np.loadtxt('Y_test.txt')
length = np.shape(x)[0]
y = np.loadtxt('Y_test_pred.txt')
print(x)
print(y)
x = tf.convert_to_tensor(x)
y = tf.convert_to_tensor(y)
with tf.Session() as sess:
    data_numpy = get_nce(x , y).eval()
    print(data_numpy)
    data_numpy /= length
    print(data_numpy)
'''