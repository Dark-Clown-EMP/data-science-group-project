from sklearn.linear_model import LinearRegression

def train_baseline_linear_model(X_train, y_train,X_test):
    model = LinearRegression()
    model.fit(X_train,y_train)
    y_pred_test = model.predict(X_test)
    y_pred_train = model.predict(X_train)
    return model, y_pred_test, y_pred_train