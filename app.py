import pandas as pd
import sklearn.model_selection
import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from flask import Flask, render_template, request
app = Flask(__name__)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/elephanthumanpredictor", methods = ["GET","POST"])
def elephanthumanpredictor():
    if request.method == "GET":
        return render_template('elephanthumanpredictor.html')
    else:
        userHeight = request.form["Height"]
        userWeight = request.form["Weight"]
        print(userHeight,userWeight)
        return KNNpredictor(float(userHeight),float(userWeight))

def KNNpredictor(height, weight):
    file = pd.read_csv("ElephantHumanDataset.txt")
    inputData = file.drop("Class",1)
    outputData = file["Class"]
    inputData = np.array(inputData)
    outputData = np.array(outputData)

    trainfeatures, testfeatures, trainclasses, testclasses = sklearn.model_selection.train_test_split(inputData,outputData, test_size=0.2)

    # print(testfeatures.shape)
    # print(trainfeatures.shape)
    # print(testfeatures[0])
    # print(testfeatures[0].shape)
    # print(trainfeatures)

    trainingPlaceholder = tf.placeholder(tf.float32, [1600,2])
    testingPlaceholder = tf.placeholder(tf.float32, [2])

    distance = tf.reduce_sum(tf.abs(trainingPlaceholder-testingPlaceholder), reduction_indices = 1)
    prediction = tf.arg_min(distance,0)
    init = tf.initialize_all_variables()

    sess = tf.Session()
    sess.run(init)
    userData = [height,weight]
    userData= np.array(userData)
    distances = sess.run(distance, {trainingPlaceholder:trainfeatures, testingPlaceholder:userData})
    # print(distances)
    predictions = sess.run(prediction, {trainingPlaceholder:trainfeatures, testingPlaceholder:userData})
    # print(predictions)
    # print(trainclasses[predictions])
    sess.close()
    return trainclasses[predictions]



if __name__ == "__main__":
    app.run()

