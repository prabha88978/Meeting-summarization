# Meeting-summarization

The aim of this project is to generate summaries for meetings.  
We built an summarization model which takes text an input and generates summary using datasets called CNN/Daily mail and NewsRoom. We trained our model on online platform called Google Colab.  
We build a local API which takes meeting's text input and generates the summary of meeting.

# Project dependencies :
* Python 3
* Flask
* NLTK
* Tensorflow
* ROUGE

# Dataset Links:
CNN/Daily mail   : https://drive.google.com/open?id=1uWoNdU7EDF1KbIivxfESzT51NEEsLVII  
NewsROom 	     : https://summari.es/download/  
Word-Embeddings  : https://drive.google.com/open?id=1qxBKLczcqA5Y682SpZhWX6Z_COrNjMDj. 

# FIle Description:   
Train the model : To train the model run "Training.ipynb"  
run the server  : execute flask_backend.py to run the API  
