import pandas as pd
import gensim, logging
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import glob, os
import json
from sklearn.manifold import TSNE

model = gensim.models.Word2Vec.load('Models/ModelMatched2.word2vec')
words = []


def display_closestwords_tsnescatterplot(model, word):
    arr = np.empty((0, 300), dtype='f')
    word_labels = [word]

    # get close words
    close_words = model.similar_by_word(word)

    # add the vector for each of the closest words to the array
    arr = np.append(arr, np.array([model[word]]), axis=0)
    for wrd_score in close_words:
        wrd_vector = model[wrd_score[0]]
        word_labels.append(wrd_score[0])
        arr = np.append(arr, np.array([wrd_vector]), axis=0)

    # find tsne coords for 2 dimensions
    tsne = TSNE(n_components=2, random_state=0)
    np.set_printoptions(suppress=True)
    Y = tsne.fit_transform(arr)

    x_coords = Y[:, 0]
    y_coords = Y[:, 1]
    # display scatter plot
    plt.scatter(x_coords, y_coords)

    for label, x, y in zip(word_labels, x_coords, y_coords):
        plt.annotate(label, xy=(x, y), xytext=(0, 0), textcoords='offset points')
    plt.xlim(x_coords.min() + 0.00005, x_coords.max() + 0.00005)
    plt.ylim(y_coords.min() + 0.00005, y_coords.max() + 0.00005)
    plt.show()


def sentence_parser(text):
    sentence_list = []
    for each_sentence in text.splitlines():
        x = each_sentence.split("/", 50)
        sentence_list += x
    return sentence_list


display_closestwords_tsnescatterplot(model, 'logosvg')


""" This part is commented for having some bugs. """

# main_data = pd.read_excel('Excel files/NewData.xlsx', sheet_name='Sheet1')
# json_str = main_data.to_json(orient="records")
#
# projects = json.loads(json_str)
#
# for each_project in projects:
#     words += sentence_parser(each_project['File structure'])
#
# words = list(dict.fromkeys(words))
#
# vector_list = [model[word] for word in words if word in model.wv.vocab]
#
# words_filtered = [word for word in words if word in model.wv.vocab]
#
# word_vec_zip = zip(words_filtered, vector_list)
#
# word_vec_dict = dict(word_vec_zip)
# df = pd.DataFrame.from_dict(word_vec_dict, orient='index')
#
# # Initialize t-SNE
# tsne = TSNE(n_components = 2, init = 'random', random_state = 10, perplexity = 100)
#
# # Use only 4000 rows to shorten processing time
# tsne_df = tsne.fit_transform(df[:3320])
#
# sns.set()
# # Initialize figure
# fig, ax = plt.subplots(figsize=(20, 16))
# sns.scatterplot(tsne_df[:, 0], tsne_df[:, 1], alpha=0.5)
#
# texts = []
# words_to_plot = list(np.arange(0, 3320, 30))
#
#
# # Append words to list
# for word in words_to_plot:
#     texts.append(plt.text(tsne_df[word, 0], tsne_df[word, 1], df.index[word], fontsize=10))
#
# # Plot text using adjust_text (because overlapping text is hard to read)
# adjust_text(texts, force_points=0.4, force_text=0.4,
#             expand_points=(2, 1), expand_text=(1, 2),
#             arrowprops=dict(arrowstyle="-", color='black', lw=0.5))
#
# print('Creating picture ...')
# plt.show()


