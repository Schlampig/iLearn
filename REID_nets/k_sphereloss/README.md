Introduction
============
A-softmax loss (in SphereFace) rather than so-called sphere loss has been introduced into the baseline model instead of softmax loss or other common loss.

---
Cite:
-----
√ Weiyang Liu, Yandong Wen, Zhiding Yu, Ming Li, Bhiksha Raj, & Le Song (2017). SphereFace : Deep Hypersphere Embedding for Face Recognition. CVPR 2017.

---
Official link:
--------------
https://github.com/wy1iu/sphereface

---
PS:
---
    1. The keras codes here to implement a-softmax loss into a model are original. Use them freely. I think there might be something wrong in it, because I never use labels information during the loss calculation.
    2. Please read introductions about baseline and center_loss to know how to use these codes 
    3. I will be glad if you could cite this link, star it or give me a feed back!
    4. Actually, I also tried ArcLoss this version:
       https://github.com/ewrfcas/Machine-Learning-Toolbox/blob/master/loss_function/ArcFace_loss.py
       but in Market-1501 dataset, it even works worse than sphereloss here... (2018.05.18.)
