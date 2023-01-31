import pandas as pd
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from keras.models import load_model

# load the CSV file into a pandas dataframe
df = pd.read_csv('main.csv')

# split the data into training and test sets
train_data = df.sample(frac=0.8, random_state=1)
test_data = df.drop(train_data.index)

# save the training and test sets to separate CSV files
train_data.to_csv('train_data.csv', index=False)
test_data.to_csv('test_data.csv', index=False)

# load the CSV file into a pandas dataframe
df = pd.read_csv('main.csv')
df = df._get_numeric_data()
print(df.columns)

# split the data into input (X) and output (y) variables
# X = df.drop('pIC50 (IC50 in microM)', axis=1)
X = df.drop('Compound No.', axis=1).astype(float)
y = df['Compound No.'].astype(float)
# y = df['output']

# split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

# create a sequential model
model = Sequential()

# add a dense layer with a single output node
model.add(Dense(1, input_dim=X_train.shape[1], activation='linear'))

# compile the model
model.compile(loss='mean_squared_error', optimizer='adam')

# train the model
model.fit(X_train, y_train, epochs=50, batch_size=32, verbose=1)

# evaluate the model on the test data
test_loss = model.evaluate(X_test, y_test, verbose=0)
print('Test loss:', test_loss)

# save the model to a h5 file
model.save('model.h5')


# load the saved model
model = load_model('model.h5')

# use the model to make predictions on new data
predictions = model.predict(X_test)
print(predictions)