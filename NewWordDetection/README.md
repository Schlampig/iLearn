### A simple new words detection algorithm for Chinese documentation.
  * Authors: [Xylander23](https://github.com/xylander23/New-Word-Detection) is the original author, and then [Lyrichu](https://github.com/Lyrichu/NewWordDetection) modified the code to python3 version.
  * Reference: [Code for Chinese Word Segmentation](https://github.com/Moonshile/ChineseWordSegmentation), [Blog about New Words Detection](http://www.matrix67.com/blog/archives/5044)

### Usage:
  * Prepare a stop-words dictionary file named [dict.txt](https://github.com/Schlampig/i_learn_deep/blob/master/NewWordDetection/dict.txt).
  * Prepare the target document file named [document.txt](https://github.com/Schlampig/i_learn_deep/blob/master/NewWordDetection/document.txt). More words, better performance, since this is a unsupervised statistic method.
  * Put both files under the same folder with the code [detect_new_words.py](https://github.com/Schlampig/i_learn_deep/blob/master/NewWordDetection/detect_new_words.py)
  * Run the code to generate a result file named ".csv".
  * Check new discovered words listed in the result file.
