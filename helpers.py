import re
import numpy as np


def find_math_words(text):
    math_words = ''
    if '\\int' in text:
        math_words += ' integral'
    if '\\lim' in text:
        math_words += ' limit'
    if '\\sum' in text:
        math_words += ' sum'
    if '\\infty' in text:
        math_words += ' infinity'
    if '{matrix}' in text or '{pmatrix}' in text or '{bmatrix}' in text:
        math_words += ' matrix'
    if '{array}' in text:
        math_words += ' array'
    if '\\exp' in text or 'e^' in text:
        math_words += ' exponential'
    if 'ln(' in text or 'log(' in text:
        math_words += ' log'
    if '\\sqrt' in text:
        math_words += ' square_root'
    if '\\frac' in text:
        math_words += ' fraction'
    if '\\sin' in text:
        math_words += ' sine'
    if '\\cos' in text:
        math_words += ' cosine'
    if '\\tan' in text:
        math_words += ' tangent'
    if '\\arctan' in text:
        math_words += ' arctangent'
    if '\\pi' in text:
        math_words += ' pi'
    if '\\partial' in text:
        math_words += ' partial'
    if '\\Delta' in text:
        math_words += ' delta'
    if '\\geq' in text or '\\leq':
        math_words += ' greater_than'
    if '\\cdot' in text:
        math_words += ' cdot'
    if '\\subset' in text or '\\subseteq' in text:
        math_words += ' subset'
    if ('\\cup' in text or '\\cap' in text
            or '\\bigcup' in text or '\\bigcap' in text):
        math_words += ' cup'
    if '\\epsilon' in text or '\\varepsilon' in text:
        math_words += ' epsilon'
    if '\\inf' in text:
        math_words += ' infimum'
    if '\\sup' in text:
        math_words += ' supremum'
    if '\\min' in text:
        math_words += ' minimum'
    if '\\max' in text:
        math_words += ' maximum'
    if '\\det' in text:
        math_words += ' determinant'
    if '^T' in text:
        math_words += ' transpose'
    if '\\mod' in text:
        math_words += ' modulo'

    return math_words


def strip_text(text):
    ''' Remove html tags, latex tags, etc. '''

    math_words = find_math_words(text)

    text = text.replace('<span class=\"math\">', 'code_word_begin')
    text = text.replace('</span>', 'code_word_end')
    text = re.sub(r'(?<=code_word_begin)(.*?)(?=code_word_end)', ' ', text,
                  flags=re.DOTALL)

    text = text.replace('<em>', 'code_word_begin')
    text = text.replace('</em>', 'code_word_end')
    text = re.sub(r'(?<=code_word_begin)(.*?)(?=code_word_end)', ' ', text,
                  flags=re.DOTALL)

    text.strip()
    text = text.lower()
    text = text.replace('<p>', ' ')
    text = text.replace('</p>', ' ')
    text = text.replace('code_word_begin', ' ')
    text = text.replace('code_word_end', ' ')
    text = text.replace('.', ' ')
    text = text.replace(',', ' ')
    text = text.replace(';', ' ')
    text = text.replace('?', ' ')
    text = text.replace('!', ' ')
    text = text.replace('\n', '')
    text = re.sub(r'[^a-z ]', ' ', text)

    text = text + math_words

    list_voc = re.split(r'[ ]+', text)

    return list_voc


def make_y_vec(data, topic_list):
    ''' Useful in multiclass regression. '''
    y_vec = np.zeros(shape=(1, len(topic_list)))

    if 'topics' in data:
        topics = data['topics']
        for topic in topics:
            topic = str(topic)
            y_vec[0][topic_list.index(topic)] = 1

    return y_vec
