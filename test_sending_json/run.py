import helpers
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import svm
import numpy as np

try:
    import cPickle as pickle
except:
    import pickle


topic_tags = ["Eigenvalues_and_eigenvectors",
              "Probability_density_function",
              "Taylor_series",
              "Substitution",
              "Lagrange_interpolation"]
pickle.dump(topic_tags, open("topics.bin", "wb"))
              
questions_raw = helpers.get_questions_with_topics(topic_tags)
X = helpers.questions_to_X(questions_raw)
y = helpers.questions_to_y(questions_raw, topic_tags)

classifier = OneVsRestClassifier(svm.SVC(kernel='linear',
                                         probability = True,
                                         random_state=np.random.RandomState(0))
 
                                )

final_classifier = classifier.fit(X, y)
pickle.dump(final_classifier, open("final.bin", "wb"))
