"""This file runs some tests on model like finding similar words and etc"""


import gensim

new_model = gensim.models.Word2Vec.load('Models/ModelMatched2.word2vec')

# print(len(new_model.wv.vocab))
#
# print(new_model.most_similar('Newcss'))

# print(new_model.doesnt_match(['newcss', 'headercss', 'Maincss', 'packagejson', 'feedcss']))

# print(str(new_model.similarity('src', 'public')))

prediction = new_model.predict_output_word('src')

print(prediction)