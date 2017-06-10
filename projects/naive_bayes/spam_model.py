import pandas as pd

# 1.1 Understanding the dataset
# ===================================================================================
# - Import the dataset into a pandas dataframe using the read_table method.
# Because this is a tab separated dataset we will be using '\t' as the value
# for the 'sep' argument which specifies this format.
# - Also, rename the column names by specifying a list ['label, 'sms_message']
# to the 'names' argument of read_table().
# - Print the first five values of the dataframe with the new column names.
# ===================================================================================

# Read from spam collection file
df = pd.read_table('smsspamcollection/SMSSpamCollection',
                    sep = '\t',
                    header = None,
                    names = ['label', 'sms_messages'])

# Take a look
# print(df.head())

# 1.2 Preprocessing
# ===================================================================================
# - Convert the values in the 'label' colum to numerical values using map
# method as follows: {'ham':0, 'spam':1} This maps the 'ham' value to 0 and
# the 'spam' value to 1.
# - Also, to get an idea of the size of the dataset we are dealing with, print
# out number of rows and columns using 'shape'.
# ===================================================================================

# Convert labels to [0, 1]
# Define mapping
mapping = {'ham' : 0, 'spam': 1}
# Apply mapping
df.label = df.label.map(mapping)

# Get an understanding of the size
# print(df.shape)

# 2.1 Bag of Worms (Sklearn)
# ===================================================================================
# Describes bag of worms, no code
# ===================================================================================

# 2.2 Bag of Worms (from scratch)
# ===================================================================================
# The example goes on a slight tangent using a smaller documents array
# I attempt this with the imported data
# ===================================================================================

# 2.2.1 Convert all strings to lower case
# ===================================================================================
# - Convert all the strings in the documents set to their lower case.
# Save them into a list called 'lower_case_documents'. You can convert
# strings to their lower case in python by using the lower() method.
# ===================================================================================

lower_case_documents = df.sms_messages.apply(lambda x: str.lower(x));

# print(lower_case_documents)

# 2.2.2 Remove all punctuation
# ===================================================================================
# - Remove all punctuation from the strings in the document set. Save
# them into a list called 'sans_punctuation_documents'.
# ===================================================================================

# Regex function
def removePunctuation(s) :
    import re, string
    regex = re.compile('[%s]+' % re.escape(string.punctuation))

    return regex.sub("", s)


sans_punctuation_documents = lower_case_documents.apply(lambda x: removePunctuation(x))

# print(sans_punctuation_documents)

# 2.2.3 Tokenization
# ===================================================================================
# - Tokenize the strings stored in 'sans_punctuation_documents' using the split()
# method. and store the final document set in a list called
# 'preprocessed_documents'.
# ===================================================================================

preprocessed_documents = sans_punctuation_documents.apply(lambda x: str.split(x))

# print(preprocessed_documents)

# 2.2.4 Count frequency
# ===================================================================================
# - Using the Counter() method and preprocessed_documents as the input, create a
# dictionary with the keys being each word in each document and the corresponding
# values being the frequncy of occurrence of that word. Save each Counter dictionary
# an item in a list called 'frequency_list'.
# ===================================================================================

from collections import Counter
frequency_list = preprocessed_documents.apply(lambda x: Counter(x))

# print(frequency_list)

# 2.3 BoW with Sklearn
# ===================================================================================
# As in 2.2, I attempt this with the imported dataset instead of a small list
# ===================================================================================

# 2.3.1 Importing
# ===================================================================================
# - Import the sklearn.feature_extraction.text.CountVectorizer
# method and create an instance of it called 'count_vector'.
# ===================================================================================
from sklearn.feature_extraction.text import CountVectorizer

count_vector = CountVectorizer()

# 2.3.2 Using the CountVectorizer
# ===================================================================================
# - Fit your document dataset to the CountVectorizer object you have created
# using fit(), and get the list of words which have been categorized as
# features using the get_feature_names() method.
# ===================================================================================

count_vector.fit(df.sms_messages)

# print(count_vector.get_feature_names())

# 2.3.3 Create the count frequency matrix
# ===================================================================================
# Create a matrix with the rows being each of the 4 documents, and the columns
# being each word. The corresponding (row, column) value is the frequency of
# occurrance of that word(in the column) in a particular document(in the row).
# You can do this using the transform() method and passing in the document data
# set as the argument. The transform() method returns a matrix of numpy integers,
# you can convert this to an array using toarray(). Call the array 'doc_array'
# ===================================================================================

doc_array = count_vector.transform(df.sms_messages).toarray()
# print(doc_array)

# 2.3.4 Convert Frequency Matrix to Dataframe
# ===================================================================================
# Convert the array we obtained, loaded into 'doc_array', into a dataframe and
# set the column names to the word names(which you computed earlier using
# get_feature_names(). Call the dataframe 'frequency_matrix'.
# ===================================================================================

columns = count_vector.get_feature_names()
frequency_matrix = pd.DataFrame(doc_array, columns = columns)

# print(frequency_matrix.head())

# 3.1 Training and testing sets
# ===================================================================================
# - Split the dataset into a training and testing set by using the train_test_split
# method in sklearn. Split the data using the following variables:
#     X_train is our training data for the 'sms_message' column.
#     y_train is our training data for the 'label' column
#     X_test is our testing data for the 'sms_message' column.
#     y_test is our testing data for the 'label' column
# - Print out the number of rows we have in each our training and testing data.
# ===================================================================================

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(df.sms_messages, df.label)

# print("Num rows in x_train: " + str(x_train.size))
# print("Num rows in x_test: " + str(x_test.size))
# print("Num rows in y_train: " + str(y_train.size))
# print("Num rows in y_test: " + str(y_test.size))
# print("Total num rows: " + str(len(df)))

# 3.2 Applying BoW to our dataset
# ===================================================================================
# - Firstly, we have to fit our training data (X_train) into CountVectorizer()
# and return the matrix.
# - Secondly, we have to transform our testing data (X_test) to return the matrix.
# ===================================================================================

cv = CountVectorizer()
training_data = cv.fit_transform(x_train)
testing_data = cv.transform(x_test)

# 4.1 Bayes Theorem From Scratch
# ===================================================================================
# Let us implement the Bayes Theorem from scratch using a simple example.
# Let's say we are trying to find the odds of an individual having diabetes,
# given that he or she was tested for it and got a positive result.
# In the medical field, such probabilies play a very important role as it usually
# deals with life and death situatuations.

# P(D) is the probability of a person having Diabetes.
#     It's value is 0.01 or in other words, 1% of the general population has diabetes.
# P(Pos) is the probability of getting a positive test result.
# P(Neg) is the probability of getting a negative test result.
# P(Pos|D) is the probability of getting a positive result on a test done for detecting
#     diabetes, given that you have diabetes. This has a value 0.9. (Sensitivity)
# P(Neg|~D) is the probability of getting a negative result on a test done for detecting
#     diabetes, given that you do not have diabetes. This also has a value of 0.9.
#     (Specificity)
# Putting our values into the formula for Bayes theorem we get:
#     P(D|Pos) = (P(D) * P(Pos|D)) / P(Pos)
#     P(Pos) = [P(D) * Sensitivity] + [P(~D) * (1-Specificity))]
# ===================================================================================

# 4.1.1
# ===================================================================================
# Calculate probability of getting a positive test result, P(Pos)
# ===================================================================================

# P(D)
p_diabetes = 0.01
# P(~D)
p_no_diabetes = 0.99
# Sensitivity P(pos|D)
p_sens = 0.9
# Specificity P(neg|~D)
p_spec = 0.9

p_pos = (p_diabetes * p_sens) + (p_no_diabetes * (1-p_spec)) # 10.8%

# print("The probabilitiy of getting a positive test result is: " + str(p_pos * 100) + "%")

# 4.1.2
# ===================================================================================
# Compute the probability of an individual having diabetes, given that, that individual
# got a positive test result. In other words, compute P(D|Pos).
# ===================================================================================

p_diabetes_given_pos = (p_diabetes * p_sens) / p_pos # 8.3%

# print("The probability of having diabetes given a positive test result is: "
#    + str(p_diabetes_given_pos * 100) + "%")

# 4.1.3
# ===================================================================================
# Compute the probability of an individual not having diabetes, given that, that individual
# got a positive test result. In other words, compute P(~D|Pos).
# P(~D|Pos) = (P(~D) * P(Pos|~D)) / P(Pos)
# ===================================================================================

p_pos_no_diabetes = 1 - p_spec # P(Pos|~D) = 1 - P(Neg|~D)
p_no_diabetes_given_pos = p_no_diabetes * p_pos_no_diabetes / p_pos # 91.67%

# print("The proability of not having diabetes given a positive test result is: "
#    + str(p_no_diabetes_given_pos * 100) + "%")

# 4.2 Naive Bayes From Scratch
# ===================================================================================
# Now that you have understood the ins and outs of Bayes Theorem, we will extend it
# to consider cases where we have more than feature.
#
# Let's say that we have two political parties' candidates, 'Jill Stein' of the
# Green Party and 'Gary Johnson' of the Libertarian Party and we have the probabilities
# of each of these candidates saying the words 'freedom', 'immigration' and 'environment'
# when they give a speech:
#       Probability that Jill Stein says 'freedom': 0.1 ---------> P(F|J)
#       Probability that Jill Stein says 'immigration': 0.1 -----> P(I|J)
#       Probability that Jill Stein says 'environment': 0.8 -----> P(E|J)
#       Probability that Gary Johnson says 'freedom': 0.7 -------> P(F|G)
#       Probability that Gary Johnson says 'immigration': 0.2 ---> P(I|G)
#       Probability that Gary Johnson says 'environment': 0.1 ---> P(E|G)
# And let us also assume that the probablility of Jill Stein giving a speech,
#       P(J) is 0.5 and the same for Gary Johnson, P(G) = 0.5.
#
# The Naive Bayes formula is P(y|x1,...,xn) = (P(Y) * P(x1,...,xn|y)) / P(x1,...,xn)
# ===================================================================================

# 4.2.1 Compute P(F,I)
# ===================================================================================
# Compute the probability of the words 'freedom' and 'immigration' being said in a
# speech, or P(F,I).
#
# The first step is multiplying the probabilities of Jill Stein giving a speech with her
# individual probabilities of saying the words 'freedom' and 'immigration'. Store this
# in a variable called p_j_text
#
# The second step is multiplying the probabilities of Gary Johnson giving a speech with
# his individual probabilities of saying the words 'freedom' and 'immigration'. Store
# this in a variable called p_g_text
#
# The third step is to add both of these probabilities and you will get P(F,I).
# ===================================================================================
# P(J) = P(G) = 0.5
p_jill = p_gary = 0.5

# P(F|J)
p_j_f = 0.1
# P(I|J)
p_j_i = 0.1
# P(F|G)
p_g_f = 0.7
# P(I|G)
p_g_i = 0.2

p_j_text = p_jill * p_j_f * p_j_i
p_g_text = p_gary * p_g_f * p_g_i

# P(F,I)
p_f_i = p_j_text + p_g_text # 7.5%

# print("P(F, I) = " + str(p_f_i))

# 4.2.2 Compute P(J|F,I)
# ===================================================================================
# Compute P(J|F,I) using the formula P(J|F,I) = (P(J) * P(F|J) * P(I|J)) / P(F,I)
# and store it in a variable p_j_fi
# ===================================================================================

p_j_fi = p_j_text / p_f_i # 6.67%

# print("The probability of the speaker being Jill, given the words 'Freedom' and 'Immigration' \
# were said is: " + str(p_j_fi * 100) + "%")

# 4.2.3 Compute P(G|F,I)
# ===================================================================================
# Compute P(G|F,I) using the formula P(G|F,I) = (P(G) * P(F|G) * P(I|G)) / P(F,I)
# and store it in a variable p_g_fi
# ===================================================================================

p_g_fi = p_g_text / p_f_i # 93.33%

# print("The probability of the speaker being Gary, given the words 'Freedom' and 'Immigration' \
# were said is: " + str(p_g_fi * 100) + "%")

# 5 Naive Bayes w/ Scikit-Learn
# ===================================================================================
# (Back to spam detection)
# We have loaded the training data into the variable 'training_data' and the testing
# data into the variable 'testing_data'.
#
# Thankfully, sklearn has several Naive Bayes implementations that we can use and so
# we do not have to do the math from scratch. We will be using sklearns
# sklearn.naive_bayes method to make predictions on our dataset.
#
# Specifically, we will be using the multinomial Naive Bayes implementation. This
# particular classifier is suitable for classification with discrete features
# (such as in our case, word counts for text classification). It takes in integer
# word counts as its input. On the other hand Gaussian Naive Bayes is better suited
# for continuous data as it assumes that the input data has a Gaussian(normal) distribution.
# ===================================================================================

# 5.1 Training
# ===================================================================================
# Import the MultinomialNB classifier and fit the training data into the classifier
# using fit(). Name your classifier 'naive_bayes'. You will be training the classifier
# using 'training_data' and y_train' from our split earlier.
# ===================================================================================

from sklearn.naive_bayes import MultinomialNB

clf = MultinomialNB()
clf.fit(training_data, y_train)

# 5.2 Predict
# ===================================================================================
# Now that our algorithm has been trained using the training data set we can now make
# some predictions on the test data stored in 'testing_data' using predict().
# Save your predictions into the 'predictions' variable.
# ===================================================================================

predictions = clf.predict(testing_data)

# 6 Evaluating Our Model
# ===================================================================================
# Now that we have made predictions on our test set, our next goal is to evaluate how
# well our model is doing. There are various mechanisms for doing so, but first let's
# do quick recap of them.
#
# Accuracy measures how often the classifier makes the correct prediction. It’s the
# ratio of the number of correct predictions to the total number of predictions
# (the number of test data points).
#
# Precision tells us what proportion of messages we classified as spam, actually were
# spam. It is a ratio of true positives(words classified as spam, and which are
# actually spam) to all positives(all words classified as spam, irrespective of whether
# that was the correct classificatio), in other words it is the ratio of
#           [True Positives/(True Positives + False Positives)]
#
# Recall(sensitivity) tells us what proportion of messages that actually were spam
# were classified by us as spam. It is a ratio of true positives(words classified as
# spam, and which are actually spam) to all the words that were actually spam,
# in other words it is the ratio of
#           [True Positives/(True Positives + False Negatives)]
#
# The Precision and REcall can be combined to get the F1 score, which is the weighted
# average of the precision and recall scores. It ranges from 0 to 1 with 1 being the
# best score
# ===================================================================================

# 6.1 Computing accuracy, precision, recall, F1
# ===================================================================================
# Compute the accuracy, precision, recall and F1 scores of your model using your
# test data 'y_test' and the predictions you made earlier stored in the 'predictions'
# variable.
# ===================================================================================
fp = tp = fn = tn = 0
for i in range(0, predictions.size):
    pi = predictions[i]
    yi = y_test.data[i]

    if (pi == yi):
        if (pi == 0): # True negative
            tn += 1
        else: # True positive
            tp += 1
    else:
        if (pi == 0): # False negative
            fn += 1
        else: # False positive
            fp += 1

accuracy = (tn + tp) / predictions.size
precision = tp / (tp + fp)
recall = tp / (tp + fn)
f1 = 2 * (precision * recall)/(precision + recall) # As stated on wikipedia

print("My attempt at manually calculating: ")
print("\tAccuracy is: " + str(accuracy))
print("\tPrecision is: " + str(precision))
print("\tRecall is: " + str(recall))
print("\tF1 score is: " + str(f1))


from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
print("SKlearn metrics: ")
print("\tAccuracy is: " + str(accuracy_score(y_test, predictions)))
print("\tPrecision is: " + str(precision_score(y_test, predictions)))
print("\tRecall is: " + str(recall_score(y_test, predictions)))
print("\tF1 score is: " + str(f1_score(y_test, predictions)))

























