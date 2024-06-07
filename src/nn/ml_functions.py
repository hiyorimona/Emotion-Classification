from sklearn.metrics import f1_score

def training(model,vect, X_train, y_train):
    X_train = vect.fit_transform(X_train)

    model.fit(X_train,y_train)
    train_preds = model.predict(X_train)
    model.get_params()
    print('F1 score on training set: {}'.format(f1_score(y_train, train_preds, average='weighted')))

    return model

def validation(model, vect, X_test, y_test):
    X_test = vect.transform(X_test)

    val_preds = model.predict(X_test)
    print('F1 score on validation set: {}'.format(f1_score(y_test, val_preds, average='weighted')))
    return val_preds

def evaluating_test(df_test, true_labels, model,vect, feature):
    df_temp = df_test.copy()
    X_test = vect.transform(true_labels[feature])

    predictions = model.predict(X_test)
    df_temp['emotion'] = predictions
    return df_temp

def sentiment_prediction(df_test, true_labels,model,vect,X_train,X_test,y_train,y_test, feature):
    trained_model = training(model, vect, X_train, y_train)
    _ = validation(trained_model, vect, X_test, y_test)
    prediction = evaluating_test(df_test, true_labels, model,vect, feature)
    print('F1 score on test set: {} \t'.format(f1_score(prediction['emotion'], true_labels['emotion'], average='weighted')))