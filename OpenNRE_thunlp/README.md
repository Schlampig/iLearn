### After transforming the format of datasets downloaded from the [BaiDu2019 Relation Extraction Competition](http://lic2019.ccf.org.cn/kg), we can try a baseline through [OpenNRE](https://github.com/thunlp/OpenNRE).
  * Create a folder named "MyPreprocess".
  * Download the 3 datasets train_data.json, dev_data.json, and all_50_schemas, and put them under a folder named "DuIE".
  * Download the [Tencent AI Lab Embedding Corpus for Chinese Words and Phraseshttps](https://ai.tencent.com/ailab/nlp/embedding.html).
  * Put "DuIE" and the Tencent embedding file under the folder MyPreprocess.
  * Create a folder under MyPreprocess named "DuNRE" to store re-formatted datasets.
  * Put the pre-process scripts gen_DuNRE.py and gen_embed_mat.py under the folder MyPreprocess.
  * Run gen_DuNRE.py to create 3 new datasets train_nre.json, dev_nre.json, and relation_nre.json.
  * Run gen_embed_mat.py to create the embedding dictionary file word_dictionary_nre.json.
  * Clone project from OpenNRE and modify the route in train_demo.py.
  * Run train_demo.py to train the model.

<br>

### Results about first 5 epochs with default settings (gpu_nums=2) for train_demo.py:
  ```
###### Epoch 1 ######
epoch 1 step 1684 time 0.04 | loss: 0.368283, not NA accuracy: 0.732326, accuracy: 0.732326
Average iteration time: 0.042350
Testing...
Calculating weights_table...
Finish calculating
[TEST] step 241 | not NA accuracy: 0.768333, accuracy: 0.766606
[TEST] auc: 0.8837323517404924
Finish testing
Best model, storing...
Finish storing
###### Epoch 2 ######
epoch 2 step 1684 time 0.05 | loss: 0.420747, not NA accuracy: 0.745189, accuracy: 0.745189
Average iteration time: 0.042155
Testing...
Calculating weights_table...
Finish calculating
[TEST] step 241 | not NA accuracy: 0.781120, accuracy: 0.779365
[TEST] auc: 0.8936706225008352
Finish testing
Best model, storing...
Finish storing
###### Epoch 3 ######
epoch 3 step 1684 time 0.05 | loss: 0.362954, not NA accuracy: 0.752812, accuracy: 0.752812
Average iteration time: 0.041912
Testing...
Calculating weights_table...
Finish calculating
[TEST] step 241 | not NA accuracy: 0.784019, accuracy: 0.782257
[TEST] auc: 0.8965463798464878
Finish testing
Best model, storing...
Finish storing
###### Epoch 4 ######
epoch 4 step 1684 time 0.05 | loss: 0.438323, not NA accuracy: 0.758082, accuracy: 0.758082
Average iteration time: 0.041687
Testing...
Calculating weights_table...
Finish calculating
[TEST] step 241 | not NA accuracy: 0.788885, accuracy: 0.787113
[TEST] auc: 0.8997719289776213
Finish testing
Best model, storing...
Finish storing
###### Epoch 5 ######
epoch 5 step 1684 time 0.04 | loss: 0.345711, not NA accuracy: 0.761369, accuracy: 0.761369
Average iteration time: 0.040897
Testing...
Calculating weights_table...
Finish calculating
[TEST] step 241 | not NA accuracy: 0.792198, accuracy: 0.790418
[TEST] auc: 0.9050112846455524
  ```

<br>

### Note:
  * Remember to modify routes of your own datasets in the OpenNRE scripts.
  * I now encounter the same issue reported [here](https://github.com/thunlp/OpenNRE/issues/103), the method proposed by [xpxpx](https://github.com/xpxpx) works. Maybe the future version of OpenNRE will address the issue.
  * ...
