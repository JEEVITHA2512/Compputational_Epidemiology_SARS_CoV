from keras.models import load_model

# load the saved model
model = load_model('model.h5')

# use the model to make predictions on new data
predictions = model.predict(X_test)