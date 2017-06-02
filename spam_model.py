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

