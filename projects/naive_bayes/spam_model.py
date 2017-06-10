import pandas as pd

# 1.1 Understanding the dataset

# - Import the dataset into a pandas dataframe using the read_table method.
# Because this is a tab separated dataset we will be using '\t' as the value
# for the 'sep' argument which specifies this format.
# - Also, rename the column names by specifying a list ['label, 'sms_message']
# to the 'names' argument of read_table().
# - Print the first five values of the dataframe with the new column names.

# Read from spam collection file
df = pd.read_table('smsspamcollection/SMSSpamCollection',
                    sep = '\t',
                    header = None,
                    names = ['label', 'sms_messages'])

# Take a look
# print(df.head())

# 1.2 Preprocessing
# - Convert the values in the 'label' colum to numerical values using map
# method as follows: {'ham':0, 'spam':1} This maps the 'ham' value to 0 and
# the 'spam' value to 1.
# - Also, to get an idea of the size of the dataset we are dealing with, print
# out number of rows and columns using 'shape'.

# Convert labels to [0, 1]
# Define mapping
mapping = {'ham' : 0, 'spam': 1}
# Apply mapping
df.label = df.label.map(mapping)

# Get an understanding of the size
# print(df.shape)

# 2.1 Bag of Worms (Sklearn)
# Describes bag of worms, no code

# 2.2 Bag of Worms (from scratch)

# 2.2.1 Convert all strings to lower case
# - Convert all the strings in the documents set to their lower case.
# Save them into a list called 'lower_case_documents'. You can convert
# strings to their lower case in python by using the lower() method.

lower_case_documents = df.sms_messages.apply(lambda x: str.lower(x));

# print(lower_case_documents)

# 2.2.2 Remove all punctuation
# - Remove all punctuation from the strings in the document set. Save
# them into a list called 'sans_punctuation_documents'.

# Regex function
def removePunctuation(s) :
    import re, string
    regex = re.compile('[%s]+' % re.escape(string.punctuation))

    return regex.sub("", s)


sans_punctuation_documents = lower_case_documents.apply(lambda x: removePunctuation(x))

# print(sans_punctuation_documents)

# 2.2.3 Tokenization
# - Tokenize the strings stored in 'sans_punctuation_documents' using the split()
# method. and store the final document set in a list called
# 'preprocessed_documents'.

preprocessed_documents = sans_punctuation_documents.apply(lambda x: str.split(x))

# print(preprocessed_documents)

# 2.2.4 Count frequency
# - Using the Counter() method and preprocessed_documents as the input, create a
# dictionary with the keys being each word in each document and the corresponding
# values being the frequncy of occurrence of that word. Save each Counter dictionary
# an item in a list called 'frequency_list'.

from collections import Counter
frequency_list = preprocessed_documents.apply(lambda x: Counter(x))

# print(frequency_list)

# 2.3 BoW with Sklearn

# 2.3.1 Importing
# - Import the sklearn.feature_extraction.text.CountVectorizer
# method and create an instance of it called 'count_vector'.
from sklearn.feature_extraction.text import CountVectorizer

count_vector = CountVectorizer()

# 2.3.2 Using the CountVectorizer
# - Fit your document dataset to the CountVectorizer object you have created
# using fit(), and get the list of words which have been categorized as
# features using the get_feature_names() method.

count_vector.fit(df.sms_messages)

# print(count_vector.get_feature_names())

# 2.3.3 Create the count frequency matrix
# Create a matrix with the rows being each of the 4 documents, and the columns
# being each word. The corresponding (row, column) value is the frequency of
# occurrance of that word(in the column) in a particular document(in the row).
# You can do this using the transform() method and passing in the document data
# set as the argument. The transform() method returns a matrix of numpy integers,
# you can convert this to an array using toarray(). Call the array 'doc_array'

doc_array = count_vector.transform(df.sms_messages).toarray()
# print(doc_array)

# 2.3.4 Convert Frequency Matrix to Dataframe
# Convert the array we obtained, loaded into 'doc_array', into a dataframe and
# set the column names to the word names(which you computed earlier using
# get_feature_names(). Call the dataframe 'frequency_matrix'.

columns = count_vector.get_feature_names()
frequency_matrix = pd.DataFrame(doc_array, columns = columns)

# print(frequency_matrix.head())

# 3.1 Training and testing sets
# - Split the dataset into a training and testing set by using the train_test_split
# method in sklearn. Split the data using the following variables:
#     X_train is our training data for the 'sms_message' column.
#     y_train is our training data for the 'label' column
#     X_test is our testing data for the 'sms_message' column.
#     y_test is our testing data for the 'label' column
# - Print out the number of rows we have in each our training and testing data.

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(df.sms_messages, df.label)

# print("Num rows in x_train: " + str(x_train.size))
# print("Num rows in x_test: " + str(x_test.size))
# print("Num rows in y_train: " + str(y_train.size))
# print("Num rows in y_test: " + str(y_test.size))
# print("Total num rows: " + str(len(df)))

# 3.2 Applying BoW to our dataset
# - Firstly, we have to fit our training data (X_train) into CountVectorizer()
# and return the matrix.
# - Secondly, we have to transform our testing data (X_test) to return the matrix.
cv = CountVectorizer()
training_data = cv.fit_transform(x_train)
testing_data = cv.transform(x_test)

# 4.1 Bayes Theorem From Scratch
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
#     diabetes, given that you do not have diabetes. This also has a value of 0.9. (Specificity)
# Putting our values into the formula for Bayes theorem we get:
#     P(D|Pos) = (P(D) * P(Pos|D)) / P(Pos)
#     P(Pos) = [P(D) * Sensitivity] + [P(~D) * (1-Specificity))]

# 4.1.1
# Calculate probability of getting a positive test result, P(Pos)

# P(D)
p_diabetes = 0.01
# P(~D)
p_no_diabetes = 0.99
# Sensitivity P(pos|D)
p_sens = 0.9
# Specificity P(neg|~D)
p_spec = 0.9

p_pos = (p_diabetes * p_sens) + (p_no_diabetes * (1-p_spec))

# print("The probabilitiy of getting a positive test result is: " + str(p_pos * 100) + "%")

# 4.1.2
# Compute the probability of an individual having diabetes, given that, that individual
# got a positive test result. In other words, compute P(D|Pos).
p_diabetes_given_pos = (p_diabetes * p_sens) / p_pos

# print("The probability of having diabetes given a positive test result is: "
#    + str(p_diabetes_given_pos * 100) + "%")

# 4.1.3
# Compute the probability of an individual not having diabetes, given that, that individual
# got a positive test result. In other words, compute P(~D|Pos).
# P(~D|Pos) = (P(~D) * P(Pos|~D)) / P(Pos)
p_pos_no_diabetes = 1 - p_spec # P(Pos|~D) = 1 - P(Neg|~D)
p_no_diabetes_given_pos = p_no_diabetes * p_pos_no_diabetes / p_pos

# print("The proability of not having diabetes given a positive test result is: "
#    + str(p_no_diabetes_given_pos * 100) + "%")





