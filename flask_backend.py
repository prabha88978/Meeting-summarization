from AbstractiveSummarization_baseline_model import testing_part
from flask import render_template
from flask import request
from flask import Flask
import os
app=Flask(__name__)

@app.route("/")
def hello():
	return render_template('index.html')

@app.route("/upload", methods=['POST'])
def upload():
	text=request.form["meeting_text"]
	print(text)
	f=open("valid.article.filter.txt","w+")
	f.write(text)
	f.close()
	a = testing_part(text)
	return render_template("summary.html",summary=a)
@app.route("/home", methods=['POST'])
def home():
	return render_template("index.html")

if __name__=="__main__":
	app.run(debug=True, host='0.0.0.0', threaded = True)