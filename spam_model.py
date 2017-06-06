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































