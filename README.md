# Meeting-summarization
Aim of this project is to generate summaries for meetings. We built an API which runs on local server.

# Requesities:
Python 3
Tensorflow
NLTK
Numpy
ROUGE (Evaluation metric)
python flask

We trained our model on Google Colab and saved the model. We have used CNN/Daily mail and News Room datasets for training.
Link for CNN/Daily mail dataset      : https://drive.google.com/open?id=1uWoNdU7EDF1KbIivxfESzT51NEEsLVII
We can download the NewsRoom dataset : https://summari.es/download/
We have used Deep Learning technique called Sequence-to-Sequence with Attention and BeamSearch for our model.
After traned these datasets we build an API in local server which takes meeting transcripts(Text) as input and generates the summary for that text. We used python flask server for back-end.
Link to download the saved model: https://drive.google.com/open?id=1DbP1Gs0zNgRFsNcBkoBenjtbwyhv4_VN
