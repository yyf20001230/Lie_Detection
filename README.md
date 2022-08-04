# Lie detection
This is a lie detection model that detects lies from video, audio, and textual input

#### 


### Additional Features

#### Speaker Identification with machine learning 
Follow the below steps to execute this feature,
  - Execute SpeakerIdentification.py file.
  - Install Required modules. 

Things to do,
  - Change all the paths in code as per your directory.
  - Four options are there when you run SpeakerIdentification.py i)Record audio for training ii)Train Model iii) Record Audio for Testing iv) Test Model(Follow the same order).
  - Store your audio recorded for training in training_set and testing audio in testing_set Folder.
  - training_set_addition.txt use this file to append trained files and testing_set_addition.txt for appending test files.
 
## Dataset

We used 60 truth and 61 deceptive videos from the testimonials dataset to train models
Videos with both processed MFCC and Action Units(AU) are in the dataset folder. 
   - MFCC annotated dataset in Wavs and AU annotated dataset in Clips 
   - For segmented for model training, access processed
   
The raw dataset can be found here: https://web.eecs.umich.edu/~mihalcea/downloads.html
Datasets labeled and annotated with Openface and Opensmile can be found here: https://drive.google.com/drive/folders/1_8pz1Te49nzkBE6FTizZfdNIt2hFBJn_?usp=sharing


## Paper Reference

[1]Feng, Kai Jiabo. DeepLie: Detect Lies with Facial Expression (Computer Vision), Standford CS230, 2021, http://cs230.stanford.edu/projects_spring_2021/reports/0.pdf

[2]Khan, Wasiq, et al. Deception in the Eyes of Deceiver: A Computer Vision and Machine Learning Based Automated Deception Detection, Science Direct, May 2021, https://doi.org/10.1016/j.eswa.2020.114341. 

[3]Shen, Xunbing, et al. Catching a Liar Through Facial Expression of Fear, Frontiers, 8 June 2021, https://www.frontiersin.org/articles/10.3389/fpsyg.2021.675097/full. 

[4]Umut S¸ en, M., and Veronica  P ´ erez-Rosas, et al. “Https://Sci-Hub.se/10.1109/TAFFC.2020.3015684.” Multimodal Deception Detection Using Real-Life Trial Data, IEEE TRANSACTIONS ON AFFECTIVE COMPUTING , 2020, https://sci-hub.se/10.1109/TAFFC.2020.3015684. 

[5]Venkatesh, Sushma, et al. Robust Algorithm for Multimodal Deception Detection, 2019 IEEE Conference, 2019, https://sci-hub.se/10.1109/MIPR.2019.00108.

[6]Zhang, Jiaxuan. Multimodal Deception Detection Using Automatically Extracted Acoustic, Visual, and Lexical Features, Columbia University, 2020, http://www.cs.hunter.cuny.edu/~slevitan/papers/interspeech_multimodal_deception_2020.pdf. 

