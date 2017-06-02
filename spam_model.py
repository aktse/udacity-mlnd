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
print(df.head())




