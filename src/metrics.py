'''
Sources:
- https://github.com/ellisdg/3DUnetCNN/blob/master/unet3d/metrics.py
'''

from functools import partial

from keras import backend as K

# https://stackoverflow.com/questions/45961428/make-a-custom-loss-function-in-keras
def binary_dice_coefficient(y_true, y_pred, smooth=1., thresh=0.5):
    '''
    y_true: a tensor with only ones and zeros.
    y_pred: a tensor that will be converted to ones and zeros using thresh
    smooth: a little extra weight for small masks and intersections
    '''
    y_pred = K.cast(y_pred > thresh, y_true.dtype)
    return dice_coefficient(y_true, y_pred, smooth=smooth)



def binary_dice_coefficient_loss(y_true, y_pred, smooth=1., thresh=0.5):
    return -binary_dice_coefficient(y_true, y_pred, smooth=smooth, thresh=thresh)


def make_dice_coefficient_loss(smooth=1e-5, smooth_numerator=True, 
                               kind='sorensen', binary=False, thresh=0.5):
    '''
    http://tensorlayer.readthedocs.io/en/latest/_modules/tensorlayer/cost.html
    kind: 'jaccard' uses (a dot a) in the denominator for length. 
      'sorensen' uses sum(a) in the denominator for length.  For binary matrices these are equivalent.
    binary: If true, convert y_true and y_pred to contain only 0's and 1's.  
      This is sometimes called the "hard" dice coefficient.
    thresh: threshold, when converting to binary, s.t. > thresh = 1, < thresh = 0.
    smooth_numerator: if 0, then if y_true is all 0, dice coef = 0, even if y_pred is all 0.  
      The hope is that during training the network will ignore y_true patches that are all 0 and still learn to
      predict zeros from the y_true patches that are not all 0.
      If > 0, then if y_true is all 0 and y_pred is all 0, dice coef = 1, which intuitively is correct.
    smooth: Avoids divide by zero issues when y_true and y_pred are both all 0.
    '''
    def dice_coefficient_loss(y_true, y_pred):
        y_true_f = K.flatten(y_true)
        y_pred_f = K.flatten(y_pred)
        if binary:
            dtype = y_true.dtype
            y_true_f = K.cast(y_true_f > thresh, dtype=dtype)
            y_pred_f = K.cast(y_pred_f > thresh, dtype=dtype)
        if smooth_numerator:
            smooth_n = smooth
        else:
            smooth_n = 0
        intersection = K.sum(y_true_f * y_pred_f)
        if kind == 'jaccard':
            denom = K.sum(y_true_f * y_true_f) + K.sum(y_pred_f * y_pred_f)
        elif kind == 'sorensen':
            denom = K.sum(y_true_f) + K.sum(y_pred_f)
        else:
            raise Exception('Unrecognized kind.', kind)
            
        return -(2. * intersection + smooth_n) / (denom + smooth)

    return dice_coefficient_loss
    

def dice_coefficient2(y_true, y_pred, smooth=1.):
    '''
    From 0 to 1.  The more overlap the better.  The less non-overlapping stuff the better.
    
    The sorensen-dice coefficient: https://en.wikipedia.org/wiki/S%C3%B8rensen%E2%80%93Dice_coefficient
    This version uses the dot product for |y_true|^2 in the denominator, not the sum.
    
    Used in V-Net. https://arxiv.org/pdf/1606.04797.pdf
    '''
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f * y_true_f) + K.sum(y_pred_f * y_pred_f) + smooth)


def dice_coefficient2_loss(y_true, y_pred):
    '''
    Higher is worse.  Minimizing the loss means maximizing the dice coefficient.
    '''
    return -dice_coefficient2(y_true, y_pred)




def dice_coefficient(y_true, y_pred, smooth=1.):
    '''
    From 0 to 1.  The more overlap the better.  The less non-overlapping stuff the better.
    The sorensen-dice coefficient: https://en.wikipedia.org/wiki/S%C3%B8rensen%E2%80%93Dice_coefficient
    This implementation takes the size of y_pred to be its sum, so it is manhattan, not euclidean.  
    The intersection is the dot product.
    '''
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def dice_coefficient_loss(y_true, y_pred):
    '''
    Higher is worse.  Minimizing the loss means maximizing the dice coefficient.
    '''
    return -dice_coefficient(y_true, y_pred)


def weighted_dice_coefficient(y_true, y_pred, axis=(-3, -2, -1), smooth=0.00001):
    """
    Weighted dice coefficient. Default axis assumes a "channels first" data structure
    :param smooth:
    :param y_true:
    :param y_pred:
    :param axis:
    :return:
    """
    return K.mean(2. * (K.sum(y_true * y_pred, axis=axis) + smooth/2) / 
                  (K.sum(y_true, axis=axis) + K.sum(y_pred, axis=axis) + smooth))


def weighted_dice_coefficient_loss(y_true, y_pred):
    return -weighted_dice_coefficient(y_true, y_pred)


def label_wise_dice_coefficient(y_true, y_pred, label_index):
    return dice_coefficient(y_true[:, label_index], y_pred[:, label_index])


def get_label_dice_coefficient_function(label_index):
    f = partial(label_wise_dice_coefficient, label_index=label_index)
    f.__setattr__('__name__', 'label_{0}_dice_coef'.format(label_index))
    return f


dice_coef = dice_coefficient
dice_coef_loss = dice_coefficient_loss


def weighted_binary_crossentropy_loss_func(w0=1, w1=1, thresh=0.5):
    '''
    Cross-entropy. Higher is worse.
    
    w0: > 0.  Used as a weight for class 0.
    w1: > 0. 
    
    Weights are normalized to lie between 0 and 1.
    
    return a Keras loss function where the cross entropy from class 0 is weighted according to w0 and 
      class1 is weighted by w1.
    '''
    # normalize
    assert w0 > 0 and w1 > 0
    
    # normalize and convert to tensors
    w0 = K.variable(w0)
    w1 = K.variable(w1)
    thresh = K.variable(thresh)
    
    def weighted_binary_crossentropy_loss(y_true, y_pred):
        '''
        This might work.
        '''
        y_true = K.flatten(y_true)
        y_pred = K.flatten(y_pred)
#         return K.mean(w1 * y_true * (-K.log(y_pred)) + w0 * (1 - y_true) * (-K.log(1 - y_pred)), axis=-1)
        return K.mean(w1 * y_true * (-K.log(y_pred)) + w0 * (1 - y_true) * (-K.log(1 - y_pred)))

    return weighted_binary_crossentropy_loss