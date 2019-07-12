# Meeting-summarization

The aim of this project is to generate summaries for meetings.  
We built an summarization model which takes text an input and generates summary.  
We trained our model using datasets called CNN/Daily mail and NewsRoom on online platform called Google Colab.  
We build a local API which takes meeting's text as input and generates the summary of meeting.

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
Afte training the model start the Flask server to open the API  To run the server : execute "flask_backend.py" then go to browser type "http://localhost:5000" , you get the interface where you can give you input(Meeting conversation in text format).  By clicking on 'click here for summary' button you get the summary of meeting.
