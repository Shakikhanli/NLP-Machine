import pandas as pd
import gensim, logging
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import glob, os
import json
from sklearn.manifold import TSNE
from adjustText import adjust_text


os.chdir("/File Structure")

model = gensim.models.Word2Vec.load('Model4.word2vec')
words = []


def sentence_parser(text):
    sentence_list = []
    for each_sentence in text.splitlines():
        x = each_sentence.split("/", 50)
        sentence_list += x
    return sentence_list


main_data = pd.read_excel('NewData.xlsx', sheet_name='Sheet1')
json_str = main_data.to_json(orient="records")

projects = json.loads(json_str)

for each_project in projects:
    words += sentence_parser(each_project['File structure'])

words = list(dict.fromkeys(words))

vector_list = [model[word] for word in words if word in model.wv.vocab]

words_filtered = [word for word in words if word in model.wv.vocab]

word_vec_zip = zip(words_filtered, vector_list)

word_vec_dict = dict(word_vec_zip)
df = pd.DataFrame.from_dict(word_vec_dict, orient='index')

# Initialize t-SNE
tsne = TSNE(n_components = 2, init = 'random', random_state = 10, perplexity = 100)

# Use only 4000 rows to shorten processing time
tsne_df = tsne.fit_transform(df[:4000])

sns.set()
# Initialize figure
fig, ax = plt.subplots(figsize=(20, 16))
sns.scatterplot(tsne_df[:, 0], tsne_df[:, 1], alpha=0.5)

texts = []
words_to_plot = list(np.arange(0, 4000, 30))


# Append words to list
for word in words_to_plot:
    texts.append(plt.text(tsne_df[word, 0], tsne_df[word, 1], df.index[word], fontsize=10))

# Plot text using adjust_text (because overlapping text is hard to read)
adjust_text(texts, force_points=0.4, force_text=0.4,
            expand_points=(2, 1), expand_text=(1, 2),
            arrowprops=dict(arrowstyle="-", color='black', lw=0.5))

print('Creating picture ...')
plt.show()
