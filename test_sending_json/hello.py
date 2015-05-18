from flask import Flask
from flask import render_template
from flask import request
from flask import jsonify
from pymongo import MongoClient
app = Flask(__name__)


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
    # Currently this simply reads the json, calculates the number of characters
    # (to prove that the data is received) and returns this information.
    # Here is where we would load the classifier, turn the content into a
    # vector and predict the topic / find similar questions.
    temp = request.get_json()
    statement = temp['statement']
    length = len(statement)
    return jsonify({"statement": statement, "length": length})

if __name__ == '__main__':
    app.run(debug=True)
