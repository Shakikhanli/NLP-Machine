import gensim

most_similar_words = []

model = gensim.models.Word2Vec.load('Models/newModel.word2vec')

for each_word in model.wv.vocab:
    print(each_word)

most_similar_words = model.most_similar(positive=['src'], topn = 10);

for x in most_similar_words:
    print(x)


