import utils
from pickle import dump, load
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.cluster.util import cosine_distance
import numpy as np
import networkx as nx

nltk.download("stopwords")
from nltk.corpus import stopwords

nltk.download("punkt")

"""
Execute till line 20 only once and then pickle the data.
(Note- Pickle files are already provided )

test_dir = '.impelsys/dataset_/dataset/stories_text_summarization_dataset_test'
train_dir = '.impelsys/dataset_/dataset/stories_text_summarization_dataset_train'
test_stories = utils.load_stories(test_dir)
train_stories = utils.load_stories(train_dir)
print('Total Test Stories %d' % len(test_stories))
print('Total Train Stories %d' % len(train_dir))

# clean stories
for example in test_stories:
	example['story'] = utils.clean_lines(example['story'].split('\n'))
	example['highlights'] = utils.clean_lines(example['highlights'])
 
for example in train_stories:
	example['story'] = clean_lines(example['story'].split('\n'))
	example['highlights'] = clean_lines(example['highlights'])

## Script to pickle files

dump(test_stories, open('.impelsys/dataset_/dataset/test_stories.pkl', 'wb'))
dump(train_stories, open('.impelsys/dataset_/dataset/train_stories.pkl', 'wb'))

"""
# Load pickle files

test_stories = load(open("test_stories.pkl", "rb"))
train_stories = load(open("train_stories.pkl", "rb"))
print("Loaded test stories %d" % len(test_stories))
print("Loaded train stories %d" % len(train_stories))

"""
The dataset have 92569 stories , as I experieced computation complexities hence considering
first 20000 articles and corresponding summaries 
"""
train_stories = train_stories[:2000]

story_ = []
summary = []
for num in train_stories:
    story_.append(num.get("story"))
    summary.append(num.get("highlights"))

print("Total articles in train data - {}".format(len(story_)))
print("Total summaries in train data - {}".format(len(summary)))

df = pd.DataFrame({"story": story_, "summary": summary})
df.head()

# Converting list to str
df["story"] = df["story"].apply(lambda x: ",".join(map(str, x)))
df["summary"] = df["summary"].apply(lambda x: ",".join(map(str, x)))

# Deriving count of words
df["story_word_count"] = df["story"].apply(lambda story: len(story))
df["summary_word_count"] = df["summary"].apply(lambda summary: len(summary))

df["story"] = df["story"].apply(
    lambda x: " ".join(
        [word for word in x.split() if word not in (stopwords.words("english"))]
    )
)

summary_with_stopwrds = df.copy()

# This data will be used for encode decoder model
# summary_with_stopwrds.to_csv('clean_data.csv')

df["summary"] = df["summary"].apply(
    lambda x: " ".join(
        [word for word in x.split() if word not in (stopwords.words("english"))]
    )
)

df["story_word_count"].plot.hist()
df["summary_word_count"].plot.hist()

"""
We could see that maximum number of words in an arctile is 80000, 
But the words with high frequency is 2000. In case of summaries , maximum nunmber of words in a summary is 350, 
But the words with high frequecy is 200 This information would be useful while training the model.
As we have to fix the length of articles and summaries
"""
