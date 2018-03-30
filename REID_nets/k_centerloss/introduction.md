Introduction
============

Center loss has been introduced into the baseline model instead of 
softmax loss or other common loss.

Note: the key code to calculate center loss is not original. It is 
copied and slightly modified from this page:

https://github.com/bojone


PS:
0. Because the test and evaluate processes are similar as 
the baseline strategy, so in this folder I only put the 
train and model file.

1. Simple loss function is easy to add into the model, keras 
constrains two input parameters and one output loss:

def my_loss(y_true, y_pre):
    # some operations
    return loss
model.compile(optimizer=xxx, loss=my_loss, metrics=['accuracy'])

2. However, trying to utilize self-designed custom loss functions 
with more than two inputs y_true and y_pre and one output loss in 
Keras is quiet difficult. One idea is to add a LossLayer into the 
model, but it might be troublesome to consider all updated parameters
during your self-defined layer. I tried and failed. Maybe I could 
handle the problem in the future.

