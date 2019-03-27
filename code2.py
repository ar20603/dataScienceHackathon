import pandas as pd
from io import StringIO
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import chi2
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
import csv


df = pd.read_csv('train_file1.csv')
col = ['Info','Category']
df = df[col]
df = df[pd.notnull(df['Info'])]
df.columns = ['Info','Category']

df['category_id'] = df['Category'].factorize()[0]
category_id_df = df[['Category', 'category_id']].drop_duplicates().sort_values('category_id')
category_to_id = dict(category_id_df.values)
id_to_category = dict(category_id_df[['category_id', 'Category']].values)


# tfidf = TfidfVectorizer(sublinear_tf=True, min_df=50, norm='l2', encoding='latin-1', ngram_range=(1, 2), stop_words='english')
# features = tfidf.fit_transform(df.Info).toarray()
# labels = df.category_id
# print(features.shape)


# N = 2
# for Category, category_id in sorted(category_to_id.items()):
# 	features_chi2 = chi2(features, labels == category_id)
# 	indices = np.argsort(features_chi2[0])
# 	feature_names = np.array(tfidf.get_feature_names())[indices]
# 	unigrams = [v for v in feature_names if len(v.split(' ')) == 1]
# 	bigrams = [v for v in feature_names if len(v.split(' ')) == 2]
	# print("# '{}':".format(Category))
	# print("  . Most correlated unigrams:\n. {}".format('\n. '.join(unigrams[-N:])))
	# print("  . Most correlated bigrams:\n. {}".format('\n. '.join(bigrams[-N:])))
# print(df.head())


X_train = df['Info']
Y_train = df['Category']
# count_vect = CountVectorizer()
# X_train_counts = count_vect.fit_transform(X_train)
# tfidf_transformer = TfidfTransformer()
# X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
# clf = MultinomialNB().fit(X_train_tfidf, Y_train)

SVC_pipeline = Pipeline([
                ('tfidf', TfidfVectorizer(stop_words=stop_words)),
                ('clf', OneVsRestClassifier(LinearSVC(), n_jobs=1)),
            ])
for category in categories:
    # print('... Processing {}'.format(category))
    # train the model using X_dtm & y
    SVC_pipeline.fit(X_train, train[category])
    # compute the testing accuracy
    prediction = SVC_pipeline.predict(X_test)
    # print('Test accuracy is {}'.format(accuracy_score(test[category], prediction)))

print(clf.predict(count_vect.transform(["History [sound recording] / Loudon Wainwright III. Charisma, p1992"])))


# f= open('test_file.csv')
# csv_f = csv.reader(f)

# data_w = open("output.csv" , "w")
# writer = csv.writer(data_w)

# for row in csv_f :
	
# 	value =""
# 	num=""
# 	ctr=0
# 	for column in row:
# 		if(ctr==0):
# 			num=column
# 		ctr+=1
# 		if column is not None :
# 			if ctr > 6:	
# 				value+=column
# 				value+=" "


# 	ans = clf.predict(count_vect.transform([value]))[0]

# 	writer.writerow([num, ans])