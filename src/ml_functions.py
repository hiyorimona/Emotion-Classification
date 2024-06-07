from sklearn.metrics import f1_score

def training(model,vect, X_train, y_train):
    """
       Trains the given model using the provided training data and vectorizer.

       Parameters:
           model: The machine learning model to be trained.
           vect: The vectorizer used to transform the input data.

           X_train (array-like): The training data features.
           y_train (array-like): The training data labels.

       Returns:
           model: The trained machine learning model.
    """
    X_train = vect.fit_transform(X_train)

    model.fit(X_train,y_train)
    train_preds = model.predict(X_train)
    model.get_params()
    print('F1 score on training set: {}'.format(f1_score(y_train, train_preds, average='weighted')))

    return model

def validation(model, vect, X_test, y_test):
    """
       Validates the given model using the provided test data and vectorizer.

       Parameters:
           model: The trained machine learning model to be validated.
           vect: The vectorizer used to transform the input data.

           X_test (array-like): The test data features.
           y_test (array-like): The test data labels.

       Returns:
           array-like: The predicted labels for the test data.
    """
    X_test = vect.transform(X_test)

    val_preds = model.predict(X_test)
    print('F1 score on validation set: {}'.format(f1_score(y_test, val_preds, average='weighted')))
    return val_preds

def evaluating_test(df_test, model,vect, feature):
    """
       Evaluates the given model on the test dataset and assigns predictions.

       Parameters:
           df_test (DataFrame): The test dataset.
           model: The trained machine learning model.
           vect: The vectorizer used to transform the input data.

       Returns:
           DataFrame: The test dataset with the predicted labels.
   """
    df_temp = df_test.copy()
    X_test = vect.transform(df_test[feature])

    predictions = model.predict(X_test)
    df_temp['emotion'] = predictions

    return df_temp

def emotion_prediction(df_test,model,vect,X_train,X_test,y_train,y_test, feature):
    """
    Trains, validates, and evaluates the model for sentiment prediction.

    Parameters:
        df_test (DataFrame): The test dataset.
        model: The machine learning model to be trained and evaluated.
        vect: The vectorizer used to transform the input data.
        feature (str): The feature column name in the test data to be used for predictions.


        X_train (array-like): The training data features.
        X_test (array-like): The test data features.
        y_train (array-like): The training data labels.
        y_test (array-like): The test data labels.

    Returns:
        None
    """
    trained_model = training(model, vect, X_train, y_train)
    _ = validation(trained_model, vect, X_test, y_test)
    prediction = evaluating_test(df_test, model, vect, feature)
    print('F1 score on test set: {} \t'.format(f1_score(prediction['emotion'], df_test['emotion'], average='weighted')))