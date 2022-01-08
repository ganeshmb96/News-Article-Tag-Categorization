**<h2>News Article Tag Categorization</h2>**

This repository contains all the research work, analysis and results obtained when developing the implementation of the paper for the parsing, identifying and categorizing the news articles in order to acquire the specific tags for each document.

The data set was obtained from https://www.kaggle.com/rmisra/news-category-dataset/metadata

I started this project out of a need to improve Search Engine Optimization for locating articles that weren't tagged. This code will help any body of text to generate tags. To understand the working read the paper tagged in the repo titled - ```News_Article_Tag_Categorization.docx```. I started this project in 2019 and hoping to continue contributing to it more often this year as I have started learning more on NLP and other advanced topics this year.

Current build stage is Alpha v1.0.0 and works for running the model on a corpus of text. If you are looking to see how GRU's, LSTM and Text-based CNN's work, then this is the repo to get started.  

Current Accuracy of the models are 
1) LSTM with Attention - 65.07%
2) Bidirectional GRU - 63.3%

2022 Goals - 
1) Working on adding in the Entity resolution code to generate tags instead of pulling it out of the bag-of-words model.
2) Build the classifier with the TF-IDF tokenization process to generate arrays of important words.
3) Launch it as a package

Here are some of the accuracies currenty for the Alpha Build

![Screenshot](https://github.com/ganeshmb96/News-Article-Tag-Categorization/blob/84c238536169ebf20aa90b4996d35963f06b3738/bigru_acc.png) ![Screenshot](https://github.com/ganeshmb96/News-Article-Tag-Categorization/blob/84c238536169ebf20aa90b4996d35963f06b3738/text_cnn_acc.png)

![Screenshot](https://github.com/ganeshmb96/News-Article-Tag-Categorization/blob/84c238536169ebf20aa90b4996d35963f06b3738/bigru_loss.png) ![Screenshot](https://github.com/ganeshmb96/News-Article-Tag-Categorization/blob/84c238536169ebf20aa90b4996d35963f06b3738/text_cnn_loss.png)

Will start accepting pull requests once beta build is out

