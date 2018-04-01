Introduction
============

Center loss has been introduced into the baseline model instead of 
softmax loss or other common loss.

---
Cite:
-----
√ Wen, Y., Zhang, K., Li, Z., & Qiao, Y.(2016). A Discriminative Feature Learning Approach for Deep Face Recognition. Computer Vision, ECCV 2016. Springer International Publishing. 

---
Note:
-----
The key code to calculate center loss is not original. It is 
copied and slightly modified from this page:

https://github.com/bojone

---
PS:
---
    0. Because both the data format and the test、 evaluate processes 
    are similar as the baseline strategy, so in this folder I only put 
    the train and model file.

    1. Simple loss function is easy to add into the model, keras 
    constrains two input parameters and one output loss:

    def my_loss(y_true, y_pre):
        # some operations
        return loss
    model.compile(optimizer=xxx, loss=my_loss, metrics=['accuracy'])

    2. However, trying to utilize self-designed custom loss functions 
    with more than two inputs y_true and y_pre and one output loss in 
    Keras is quite difficult.   
    One idea is to add a LossLayer into the model, just like this:
           https://github.com/Peter554/MNIST_center_loss
    but it might be troublesome to consider all updated parameters
    during your self-defined layer. I tried and failed. Maybe I could 
    handle the problem in the future.

    3. According to https://spaces.ac.cn/archives/4493, if you use Embedding 
    layer to store center and use sparse_categorical_crossentropy as the loss, 
    you don't need to transform the input label to one-hot form anymore, just 
    input the simple numeric encoded label into the net!

    However, if you insist on passing one-hot encoded label into the net to make
    it more familiar like some general models, you could make small modified in 
    the get_generator() function and the optimizer setting of the model like this:

    def get_generator(gen, class_num, batch_size):
        while True:
            X, y = gen.next()  # X with shape (batch_size, dimension), y with shape (batch_size, one_hot)
            y_pre = np.random.randint(0, class_num, y.shape)  # y_pre with shape (batch_size,)
            new_y = np.argwhere(y == 1)[:, 1]  # y_new with shape (batch_size,)
            yield [X, new_y], [y, y_pre]  # [data1, data2, ...], [label1, label2, ...]

    model.compile(optimizer=optimizer,
                  loss=['categorical_crossentropy', lambda y_true, y_pred: y_pred],
                  loss_weights=[1., 0.5],
                  metrics={'dense_output': 'accuracy'})

    rather than this:

    def get_generator(gen, class_num, batch_size):
        while True:
            X, y = gen.next()  # X with shape (batch_size, dimension), y with shape (batch_size, one_hot)
            y_pre = np.random.randint(0, class_num, y.shape)  # y_pre with shape (batch_size,)
            new_y = np.argwhere(y == 1)[:, 1]  # y_new with shape (batch_size,)
            yield [X, new_y], [new_y, y_pre]  # [data1, data2, ...], [label1, label2, ...]

    model.compile(optimizer=optimizer,
                  loss=['sparse_categorical_crossentropy', lambda y_true, y_pred: y_pred],
                  loss_weights=[1., 0.001],
                  metrics={'dense_output': 'accuracy'})
                      
    According to my experiments on Market1501, the latter is better.
    Further, I still have no idea why opimization with SGD performs better than Adam.
