from flask import Flask
from flask import render_template
from flask import request
from flask import jsonify
from pymongo import MongoClient
app = Flask(__name__)
#from sklearn.multiclass import OneVsRestClassifier
#from sklearn.linear_model import LogisticRegression
#from sklearn import svm
#import numpy as np

try:
    import cPickle as pickle
except:
    import pickle
    
import helpers


@app.route('/')
def hello_world():
    '''
    Front page.
    Finds all questionIDs so that they can be chosen from drop down.
    '''
    client = MongoClient()
    db = client.merdb
    questions = db['questions']
    all_IDs = []
    for q in questions.find():
        try:
            all_IDs.append({'id': q['ID'], 'statement': q['statement_html']})
        except KeyError:
            pass
    return render_template('web.html', questionIDs=all_IDs)


@app.route('/result', methods=['GET', 'POST'])
def result():
    topic_tags = pickle.load(open("topics.bin", "r"))
    temp = request.get_json()
    
    # THE BELOW CODE WAS RUN IN RUN.PY TO CREATE final_classifier
    #questions_raw = helpers.get_questions_with_topics(topic_tags)
    #X = helpers.questions_to_X(questions_raw)
    #y = helpers.questions_to_y(questions_raw, topic_tags)

    #classifier = OneVsRestClassifier(svm.SVC(kernel='linear',
     #                                    probability = True,
      #                                   random_state=np.random.RandomState(0))
 
       #                         )

    #final_classifier = classifier.fit(X, y)
    #!!!
    
    final_classifier = pickle.load(open("final.bin", "r"))  # - doesn't work
    
    statement = temp['statement']
    
    q = helpers.find_question(statement)
    pred = helpers.determine_topic_for_question(q, final_classifier, topic_tags)
    return jsonify({"statement": statement, "ptopic": helpers.beautify(pred)})

if __name__ == '__main__':
    app.run(debug=True)
