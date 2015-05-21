from flask import Flask
from flask import render_template
from flask import request
from flask import jsonify
from pymongo import MongoClient
app = Flask(__name__)

#more imports
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
    topic_tags = ["Eigenvalues_and_eigenvectors",
              "Probability_density_function",
              "Taylor_series",
              "Substitution", "Lagrange_interpolation"]
    # Currently this simply reads the json, calculates the number of characters
    # (to prove that the data is received) and returns this information.
    # Here is where we would load the classifier, turn the content into a
    # vector and predict the topic / find similar questions.
    temp = request.get_json()
    final_classifier = pickle.load(open("final.bin", "r"))
    
    statement = temp['statement']
    
    q = helpers.find_question(statement)
    pred = helpers.determine_topic_for_question(q, final_classifier, topic_tags)
    return jsonify({"statement": statement, "ptopic": helpers.beautify(pred)})

if __name__ == '__main__':
    app.run(debug=True)
