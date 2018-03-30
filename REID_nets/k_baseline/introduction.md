Introduction
=============

This keras version is re-written according to the basic pytorch one from the following page:
--------------------------------------------------------------------------------------------
https://github.com/layumi/Person_reID_baseline_pytorch

Please note: 
------------
    1.My version is simpler and might not perform as well as the original pytorch version, 
    e.g., I didn't use kaiming distance or learning rate schecdul. 
    2. Moreover, I use sklearn to calculated AP(Area under the Precision and Recall Curve), 
    which might make the mAP score different from the original results.
    3.According to my experiments, 
    Rank1(k_top=1), mAP(k_neighbor=1), Rank5(k_top=5) is 69.2101, 67.0975, 82.0466 for DukeMTMC, 
                                                  for Market1501.

---
Links might be useful:
----------------------
1. Datasets
http://robustsystems.coe.neu.edu/sites/robustsystems.coe.neu.edu/files/systems/projectpages/reiddataset.html
2. Datasets and projects
http://www.liangzheng.org/

---
The organization of folders to load datasets and store results is listed below:
-------------------------------------------------------------------------------
    
    datasets|
    --------|Market1501
            |DukeMTMC
            |MARS
            |……
            --------|train
                    |val
                    |gallery
                    |query
                    --------|0001
                            |0002
                            |0003
                            |……
                            --------|0001_c1_23kn332.jpg
                                    |0001_c2_3kne83n.jpg
                                    |……
    code_file|codes
             |results|
             --------|trained_model.h5
                     |feature_gallery_samples.mat
                     |results_for_the_model.mat

Please note: 
-------------
    1. I only use one sample of each class in validation set.
    2. Considering reid an open-set problem, samples in training and validation sets should not appear in gallery and query sets.
    3. Samples in query could generated from gallery set, but when you do the evaluation, samples with same camera should be filtered.
    4. Usually, the open sources datasets have split train, gallery, and query data.
        
