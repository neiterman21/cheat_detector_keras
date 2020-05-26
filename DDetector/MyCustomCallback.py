from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
#from DeceptionDetector import printPref

def printPref(pred,data_label_test):
    print("printing total preformans")
    full_mat = [[0,0],[0,0]]
    for p , a in zip(pred, data_label_test):
        if p == 0 and a == 0:
            full_mat[0][0] += 1
        if p == 0 and a == 1:
            full_mat[1][0] += 1
        if p == 1 and a == 0:
            full_mat[0][1] += 1
        if p == 1 and a == 1:
            full_mat[1][1] += 1

    print("full mat: ")
    print (full_mat[0][0] , full_mat[0][1])
    print (full_mat[1][0] , full_mat[1][1])


class MyCustomCallback(tf.keras.callbacks.Callback):

  def  __init__(self, test_data, lable):
    self.val_data = test_data
    self.lable = lable

  def on_epoch_end(self, epoch, logs=None):

    pred = self.model.predict_classes(self.val_data, verbose=2)
    printPref(pred,self.lable)
