import tensorflow as tf
import keras
from sklearn.metrics import auc, roc_curve
import numpy as np

class PreciseAUC(keras.callbacks.Callback):

    def __init__(self, *args):
        super().__init__()
        self.x_train, y_train, self.x_test, y_test = args
        self.y_train = np.argmax(y_train, axis=1)
        self.y_test = np.argmax(y_test, axis=1)

    def _compute_auc(self, y_true, y_pred):
        fpr, tpr, _ = roc_curve(y_true, y_pred)
        return auc(fpr, tpr)


    def on_epoch_end(self, epoch, logs):

        keys = ['precise_train_auc', 'precise_test_auc']
        for key in keys:
            if key not in logs:
                logs[key] = []

        y_train_pred = self.model.predict_proba(self.x_train)[:, 1]
        logs['precise_train_auc'].append(self._compute_auc(self.y_train, y_train_pred))

        y_test_pred = self.model.predict_proba(self.x_test)[:, 1]
        logs['precise_test_auc'].append(self._compute_auc(self.y_test, y_test_pred))

        print('train auc:', logs['precise_train_auc'][-1])
        print('test auc:', logs['precise_test_auc'][-1])



# define roc_callback, inspired by https://github.com/keras-team/keras/issues/6050#issuecomment-329996505
def auc_roc(y_true, y_pred):
    # any tensorflow metric
    value, update_op = tf.contrib.metrics.streaming_auc(y_pred, y_true)

    # find all variables created for this metric
    metric_vars = [i for i in tf.local_variables() if 'auc_roc' in i.name.split('/')[1]]

    # Add metric variables to GLOBAL_VARIABLES collection.
    # They will be initialized for new session.
    for v in metric_vars:
        tf.add_to_collection(tf.GraphKeys.GLOBAL_VARIABLES, v)

    # force to update metric values
    with tf.control_dependencies([update_op]):
        value = tf.identity(value)
        return value