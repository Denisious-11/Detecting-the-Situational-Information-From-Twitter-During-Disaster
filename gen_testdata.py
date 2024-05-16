import pandas as pd

my_df=pd.read_csv('Dataset/tweets_data.csv')

df= my_df[['text', 'target']]
# Define the number of rows for each class
num_rows_per_class = 100

# Create subsets for each class
non_s_subset = df[df['target'] == 0].tail(num_rows_per_class)
s_subset = df[df['target'] == 1].tail(num_rows_per_class)

# Concatenate the subsets to create the new dataset
new_df = pd.concat([non_s_subset, s_subset], ignore_index=True)

# Shuffle the rows to randomize the order
new_df = new_df.sample(frac=1).reset_index(drop=True)

# Save the new dataset to a CSV file
new_df.to_csv('test_data.csv', index=False)# testing 