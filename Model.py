print("--- Importing Packages ---")
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import jaccard_score

print("--- Load/Process Data ---")
df = pd.read_csv("aud/aud_final_proj/data.csv", header=None)
df.columns = ('category', 'text', 'postid')

# Convert Categories to numerical values
id = 1
seen = {}
for idx, row in df.iterrows():
    if row.category == 'resumes / job wanted':
        df.loc[idx, 'category'] = 0
    else:
        if row.category in seen:
            df.loc[idx, 'category'] = seen[row.category]
        else:
            df.loc[idx, 'category'] = id
            seen[row.category] = id
            id += 1

job_list = df[df["category"] != 0].copy()
resume_list = df[df["category"] == 0].copy()

print("--- Tokenization ---")
# Lemmatize, remove stop words and punctuation
processed_collection_r = []
processed_collection_j = []
lemmatizer = nltk.stem.WordNetLemmatizer()

print("Resumes...")
for post in resume_list.text:
    tokens = nltk.word_tokenize(post)
    tokens = [lemmatizer.lemmatize(token.lower()) for token in tokens]
    tokens = [token for token in tokens if not token in stopwords.words('english') if token.isalpha()]
    joins = " ".join(tokens)
    processed_collection_r.append(joins)

print("Jobs...")
for post in job_list.text:
    tokens = nltk.word_tokenize(post)
    tokens = [lemmatizer.lemmatize(token.lower()) for token in tokens]
    tokens = [token for token in tokens if not token in stopwords.words('english') if token.isalpha()]
    joins = " ".join(tokens)
    processed_collection_j.append(joins)

X = processed_collection_j
y = list(job_list.category)

print("--- Train/Test Data ---")
#Split the data into training and test
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=123)

print("--- Vectorization ---")
#Vectorize train and test data
vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=1)
# vectorizer = CountVectorizer(ngram_range=(1, 2), min_df=1)
vectorizer.fit(x_train)
x_train_m = vectorizer.transform(x_train)
x_test_m = vectorizer.transform(x_test)

print("--- Building Models ---")
print("< Naive Bayes >")
#Naive Bayes Model
nb = MultinomialNB()
nb.fit(x_train_m, y_train)
y_pred_nb = nb.predict(x_test_m)
acc_nb = accuracy_score(y_test, y_pred_nb)
print("NB Accuracy: {}%".format(round(acc_nb*100,2)))

#SVM Model
print("< SVM >")
# svm = make_pipeline(StandardScaler(with_mean=False), LinearSVC(random_state=123))
svm = LinearSVC(max_iter=2000, random_state=47906)
svm.fit(x_train_m, y_train)
y_pred_svm = svm.predict(x_test_m)
acc_svm = accuracy_score(y_test, y_pred_svm)
print("SVM Accuracy: {}%".format(round(acc_svm*100, 2)))

print("< Decision Tree >")
# Decision Tree Model
dt = DecisionTreeClassifier(max_depth=50, random_state=47906)
dt.fit(x_train_m, y_train)
y_pred_dt = dt.predict(x_test_m)
acc_dt = accuracy_score(y_test, y_pred_dt)
print("Decision Tree Accuracy: {}%".format(round(acc_dt*100, 2)))

print("< Random Forest >")
#Random Forest Model
rf = RandomForestClassifier(n_estimators=75, max_depth=50, bootstrap=True, random_state=47906)
rf.fit(x_train_m, y_train)
y_pred_rf = rf.predict(x_test_m)
acc_rf = accuracy_score(y_test, y_pred_rf)
print("RF Accuracy: {}%".format(round(acc_rf*100, 2)))

print("< Neural Network >")
#Neural Network Model
nn = MLPClassifier(solver='sgd', hidden_layer_sizes=(5,25,20), max_iter=1000, random_state=47906)
nn.fit(x_train_m, y_train)
y_pred_nn = nn.predict(x_test_m)
acc_nn = accuracy_score(y_test, y_pred_nn)
print("NN Accuracy: {}%".format(round(acc_nn*100, 2)))

print("--- Calculating Predictions ---")
# Use SVM for prediction
resume_vec = vectorizer.transform(processed_collection_r)
y_resume = svm.predict(resume_vec)

# Decode predictions and write into csv file
categories = {v: k for k, v in seen.items()}

print("Writing predictions to 'predicted_categories.csv'...")
resume_list.loc[:,'pred_category'] = y_resume
# resume_list.pred_category = resume_list.pred_category.astype('object')
resume_list.to_csv("predicted_categories.csv")

# Vectorize resume and job lists and save it into dictionary with its categories
print("--- Divide by Category ---")
idx_v = {}
for idx, row in resume_list.iterrows():
    resume_list.loc[idx, 'pred_category_text'] = categories[row.pred_category]
    post = row.text
    id = row.postid
    tokens = nltk.word_tokenize(post)
    tokens = [lemmatizer.lemmatize(token.lower()) for token in tokens]
    tokens = [token for token in tokens if not token in stopwords.words('english') if token.isalpha()]
    joins = " ".join(tokens)
    v = vectorizer.transform([joins])
    if row.pred_category in idx_v:
        idx_v[row.pred_category].append((v, id))
    else:
        idx_v[row.pred_category] = [(v, id)]

jobs_idx_v = {}
for idx, row in job_list.iterrows():
    post = row.text
    id = row.postid
    tokens = nltk.word_tokenize(post)
    tokens = [lemmatizer.lemmatize(token.lower()) for token in tokens]
    tokens = [token for token in tokens if not token in stopwords.words('english') if token.isalpha()]
    joins = " ".join(tokens)
    v = vectorizer.transform([joins])
    if row.category in jobs_idx_v:
        jobs_idx_v[row.category].append((v, id))
    else:
        jobs_idx_v[row.category] = [(v, id)]


print("--- Top Resumes by Job ---")
#Predict top resumes that match each job using document comparasion
pred_resumes = pd.read_csv("predicted_categories.csv")

jobs_matching_resumes = {}

for job_category in jobs_idx_v:
    jobs = jobs_idx_v[job_category][:3]
    for job_vtext, job_id in jobs:
        v = job_vtext.toarray()
        if job_category in idx_v:
            v_text = idx_v[job_category]
            top = None
            sims = []
            for text, id in v_text:
                sim = cosine_similarity([v[0, :]], [text.toarray()[0, :]])
                #                 sim = jaccard_score(v[0,:],text.toarray()[0,:], average='weighted')
                #                 print(sim)
                sims.append((sim, id))

            sims.sort(key=lambda x: x[0], reverse=True)
            jobs_matching_resumes[job_id] = sims[:3]

#Save results into csv file
print("Writing similarities to 'jobs_top_resumes.csv'...")
f = open('jobs_top_resumes.csv', 'w')
f.write('job_id,job_text,res1_id,res1_text,res2_id,res2_text,res3_id,res3_text\n')
for job_id in jobs_matching_resumes:
    job_text = list(df[df['postid'] == job_id].text)[0].replace(',','')
    f.write("{},{}".format(job_id, job_text))
    for resume in jobs_matching_resumes[job_id]:
        if resume[0] > 0.04:
            resume_id = resume[1]
            resume_text = list(df[df['postid'] == resume_id].text)[0].replace(',','')
            f.write(",{},{}".format(resume_id,resume_text))
    f.write('\n')
f.close()

print("--- Top Jobs by Resume ---")
#Predict top jobs that matche each resume using document comparasion
resumes_matching_jobs = {}
for category in idx_v:
    potential_list = jobs_idx_v[category]
    for resume_vtext, resume_id in idx_v[category]:
        resume_vtext = resume_vtext.toarray()

        sims = []
        for job_vtext, job_id in potential_list:
            job_vtext = job_vtext.toarray()

            sim = cosine_similarity([resume_vtext[0, :]], [job_vtext[0, :]])
            sims.append((sim[0][0], job_id))

        sims.sort(key=lambda x: x[0], reverse=True)
        resumes_matching_jobs[resume_id] = sims[:3]

#Save results into csv file
print("Writing similarities to 'resume_top_jobs.csv'...")
f = open('resume_top_jobs.csv', 'w')
f.write('res_id,res_text,job1_id,job1_text,job2_id,job2_text,job3_id,job3_text\n')
for res_id in resumes_matching_jobs:
    res_text = list(df[df['postid'] == res_id].text)[0].replace(',','')
    f.write("{},{}".format(res_id, res_text))
    for job in resumes_matching_jobs[res_id]:
        if job[0] > 0.04:
            job_id = job[1]
            job_text = list(df[df['postid'] == job_id].text)[0].replace(',','')
            f.write(",{},{}".format(job_id,job_text))
    f.write('\n')
f.close()