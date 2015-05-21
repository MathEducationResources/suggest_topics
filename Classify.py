
# coding: utf-8

# # Predict the topic of a Math Question on Math Education Resources

# We will use **Machine Learning** to predict the topic of a Math Question from the [Math Education Resources](http://math-education-resources.com). For simplicity we will only consider two topics. Using [multiclass classification](https://en.wikipedia.org/wiki/Multiclass_classification) this can be extended to more than two topics (at the time of writing, April 2015, we have about 1500 questions with 150 topics on MER).

# To Do:
# 
# 1. ~~Clean up the code (move helper functions to helper.py) - Bernhard~~
# 2. Fix pca; get feature importance - Alex
# 3. Write convenience functions:
#   1. text -> topic
#   2. text -> list of most similar questions (k-nn / cosine dist)  - Alex
# 4. Add the suggested topics to the database for questions w/o a topic
# 5. ~~Re-write code for parent topics - Bernhard~~ -> `question_to_parents` now available
# 6. ~~Re-write train test split to:  - Alex~~
#     ~~1. get at least one question from each topic~~
#     ~~2. pick them with diff probabilities~~~~
#       3. make the code look good / account for errors - Alex
# 7. ROC curve - fix for unbalanced data - Alex
# 8. Put classifier predictions on the website - Alex
# 9. Edit functions to work for both regular and parent topics - Bernhard 
# 
# -----------------------
# For later:
# 7. Add additional features (course, etc.)
#     1. graph them
# 8. Put up recommendations on the website

# In[17]:

import os
import json
import numpy as np
import helpers
from pymongo import MongoClient
import matplotlib.pyplot as plt
get_ipython().magic(u'matplotlib inline')
import pickle

# machine learning modules
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D


# In[18]:

# create an array of all topics of interest
topic_tags = ["Eigenvalues_and_eigenvectors",
              "Probability_density_function",
              "Taylor_series",
              "Substitution", "Lagrange_interpolation"]


questions_raw = helpers.get_questions_with_topics(topic_tags)

print('TOTAL:\n## - Topic\n==========')
for topic, count in helpers.count_topics_in_questions(questions_raw).iteritems():
    print('%2d - %s' %(count, topic.replace('_', ' ')))


# In[20]:

client = MongoClient()
questions_collection = client['merdb'].questions
topics_collection = client['merdb'].topics

def get_topic_to_parent_dict():
    '''returns dict topic -> parent_topic'''
    topic_to_parent_dict = dict()
    for q in topics_collection.find():
        topic_to_parent_dict[q['topic']] = q['parent']
    return topic_to_parent_dict

topic_to_parent_dict = get_topic_to_parent_dict()

def topic_to_parent(topic):
    '''returns parent for given topic'''
    return topic_to_parent_dict[topic]

def question_to_parents(q):
    '''returns sorted list of all unique parents of the questions, or [None] if question has no topics or topic is unknown.'''
    if not 'topics' in q.keys():
        return [None]
    parents = []
    for topic in q['topics']:
        parents.append(topic_to_parent(topic))
    return sorted(list(set(parents)))

def questions_to_parents(qs):
    '''return list of sorted list of all unique parents for all questions.'''
    list_of_parents = []
    for q in qs:
        list_of_parents.append(question_to_parents(q))
    return list_of_parents

questions_to_parents(questions_raw)


# ### Split into train and test set

# In[21]:

def question_indices_by_topic(qs):
    '''Returns the list of len(topic_tags) containing a list of question indices for each topic'''
    all_indices = [[] for i in range(len(topic_tags))]
    for i, q in enumerate(qs):
        for j, t in enumerate(topic_tags):
            if t in q['topics']:
                all_indices[j].append(i)
    return all_indices
                

for i, topic in enumerate(topic_tags):
    print "Question indices for topic %s: \n" % topic, question_indices_by_topic(questions_raw)[i]
    print "---------------------------------------------------------------------------"


# In[22]:

# for reproducibility we set the seed of the random number generator
np.random.seed(23)

def pick_random_index_per_topic(qs):
    '''Returns a list of randomly chosen question indices - one for each topic'''
    question_indices = question_indices_by_topic(qs)
    result = []
    for indices in question_indices:
        # pick random index
        question_index_for_topic = np.random.choice(indices)
        # add to result list, avoiding duplicates in case questions match more than one topic
        if question_index_for_topic not in result:
            result.append(question_index_for_topic)
    return result

print "Questions picked:", pick_random_index_per_topic(questions_raw)


# In[23]:

def remove_from_question_indices(ls, indices_by_topic):
    '''Takes a list ls and a list of lists indices_by_topic
        and removes all elements of ls from each element of indices_by_topic'''
    
    for index_list in indices_by_topic:
        for element in ls:
            if element in index_list:
                index_list.remove(element)
    
    return indices_by_topic

#example
print(remove_from_question_indices([1, 2, 3], [[1, 2, 4],[4, 3],[],[5]]))


# In[24]:

# helper functions for the train_test_split function

def sample_from_all_classes(indices_by_topic, num_total_samples, num_questions):
    if (num_total_samples <= 0):
        return []
    
    sample_indices = set([])
    for index_list in indices_by_topic:
        class_proportion = float(len(index_list)) / num_questions
        num_class_samples = int(num_total_samples * class_proportion)
        class_samples = sample_from_class(index_list, num_class_samples)
        # update the set sample_indices with new class samples
        sample_indices.update(class_samples)
    return list(sample_indices)

def sample_from_class(indices, n):
    return np.random.choice(indices, n, replace = False)


# In[25]:

np.random.seed(23)

def train_test_split(qs, TRAIN_PROPORTION=0.75):
    '''randomly splits list of questions into two lists for train and test'''
    TRAIN_SIZE = int(TRAIN_PROPORTION * len(qs))
                        
    # pick a question from each topic and add to training set
    indices_from_each_topic = pick_random_index_per_topic(qs)
                        
    # from the rest of the questions, pick indices from each class according to topic probabilities:
    indices_left = remove_from_question_indices(indices_from_each_topic, question_indices_by_topic(qs))
    samples_left_to_take = TRAIN_SIZE - len(indices_from_each_topic)
    
    randomly_picked_indices = sample_from_all_classes(indices_left, 
                                                    samples_left_to_take, 
                                                     len(qs)-len(indices_from_each_topic))
    
    train_indices = indices_from_each_topic + randomly_picked_indices
    
   
    qs_train = [q for i, q in enumerate(qs) if i in train_indices]
    qs_test = [q for i, q in enumerate(qs) if not i in train_indices]
    
    permuted = np.random.permutation(len(qs_train))
    qs_train_permuted = [qs_train[i] for i in permuted]
    return qs_train_permuted, qs_test

questions_train, questions_test = train_test_split(questions_raw)

print('TRAIN/TEST:\n##/## - Topic\n=============')
for t in topic_tags:
    print('%2d/%2d - %s' % (sum([1 for q in questions_train if t in q['topics']]),
                          sum([1 for q in questions_test if t in q['topics']]),
                          t.replace('_', ' ')))    


# In[27]:

vectorizer = helpers.save_TfidfVectorizer(questions_train)

X_train = helpers.questions_to_X(questions_train)
X_test = helpers.questions_to_X(questions_test)
assert X_train.shape[0] == len(questions_train)

y_train = helpers.questions_to_y(questions_train, topic_tags)
y_test = helpers.questions_to_y(questions_test, topic_tags)
assert len(y_train) == len(questions_train)


# ### The actual classifier

# In[28]:

# SVC for now
classifier = OneVsRestClassifier(svm.SVC(kernel='linear',
                                         probability = True,
                                         random_state=np.random.RandomState(0))
                                )
trained_classifier = classifier.fit(X_train, y_train)
pickle.dump(trained_classifier, open("svc.bin", "wb"))


# In[30]:

preds = trained_classifier.predict_proba(X_test)
predicted_classes = helpers.preds_to_topics(preds, topic_tags)

print('\n'.join(predicted_classes))


# In[29]:

print('%.5f combined micro AUC score.' %helpers.combined_roc_score(y_test, preds)[0])


# ## Visualize (todo)

# In[32]:

pca = PCA(n_components=3)
pca.fit(X_train.toarray())
pca_X_train = pca.transform(X_train.toarray())
pca_X_test = pca.transform(X_test.toarray())
print('The first 3 principal components explain %.2f of the variance in the dataset.' % sum(pca.explained_variance_ratio_))


# In[ ]:

#labels_train = [TOPIC1 if _ else TOPIC0 for _ in y_train]
#labels_test = [TOPIC1 if _ else TOPIC0 for _ in y_test]
fig = plt.figure(1, figsize=(8, 6))
plt.clf()
ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=25, azim=70)
for c, i, label in zip('rgb', class_indices, labels_train):
    ax.scatter(pca_X_train[y_train == i, 0],
               pca_X_train[y_train == i, 1],
               pca_X_train[y_train == i, 2],
               c=c, label=label)
    
    
for c, i, label in zip('rgb', [0, 1], [l + ' (test)' for l in labels_test]):
    ax.scatter(pca_X_test[y_test == i, 0],
               pca_X_test[y_test == i, 1],
               pca_X_test[y_test == i, 2],
               c=c, label=label, marker='x')
plt.legend()
plt.show()


# In[ ]:

fig = plt.figure(1, figsize=(8, 6))
plt.clf()
ax = Axes3D(fig, rect=[0, .5, .4, 1], elev=25, azim=70)

y_index_train = questions_to_topic_index(questions_train)

print(np.random.rand(num_classes,))

for col, i in zip(np.random.rand(num_classes,), range(num_classes)):
    print(col)
    ax.scatter(pca_X_train[y_index_train==i,0],
           pca_X_train[y_index_train==i,1],
           pca_X_train[y_index_train==i,2],
          c=col, label = topic_tags[i])
plt.legend()
plt.show()

## fix colours (e.g. through random number generator mapped to random cols)


# In[121]:

def predict_topic_for_question(q, classifier, voc):
    vec = question_to_vector(q, voc)
    pred_prob = classifier.predict_proba(vec)
    pred_class = pred_to_topic(pred_prob)
    return pred_class
    


# In[124]:

print(predict_topic_for_question(questions_raw[77], trained_classifier, vocabulary_sorted))


# In[ ]:



