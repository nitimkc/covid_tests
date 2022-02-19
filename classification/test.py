from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier


cols = ['cough', 'fever', 'sore_throat', 'shortness_of_breath', 'head_ache',
'Gender', 'Gender_1', 'Sixties', 'Sixties_1', 'Test_Reason', 'Test_Reason_1',
'Num_Tests_7dMA', 'Ave_Symptomatic_Tests_7dMA', 'Ave_Pos_Asymp_7d_MA',
'Ave_Pos_Sympt_7dMA', 'Ave_Pos_Rate_7dMA',]
       
# lr = LogisticRegression(penalty='none', fit_intercept=False, solver='lbfgs', tol=1e-2)
lr = MLPClassifier()
lr.fit(X_train[cols], y_train)
y_pred_prob = lr.predict_proba(X_test[cols])
y_pred_valid_prob = lr.predict_proba(X_valid[cols])

for i,j in zip(X_train[cols], lr.coef_[0]):
    print(i,j)
print(f'AUC test: {roc_auc_score(y_test, y_pred_prob[:, 1])}')
print(f'AUC valid: {roc_auc_score(y_valid, y_pred_valid_prob[:, 1])}')

# for i in X_train.columns:
#     print(X_train[i].value_counts())