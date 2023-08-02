# https://towardsdatascience.com/machine-learning-with-python-classification-complete-tutorial-d2c99dc524ec

## for data
import pandas as pd
import numpy as np

## for plotting
import matplotlib.pyplot as plt
import seaborn as sns

## for statistical tests
import scipy
import statsmodels.formula.api as smf
import statsmodels.api as sm

## for machine learning
from sklearn import model_selection, preprocessing, feature_selection, ensemble, linear_model, metrics, decomposition## for explainer
from lime import lime_tabular

import datetime

CSV_INPUT = 'work.keep.csv'
#CSV_INPUT = 'work.fake.csv'

FILE_LOG = 'log.txt'

'''
Recognize whether a column is numerical or categorical.
:parameter
    :param dtf: dataframe - input data
    :param col: str - name of the column to analyze
    :param max_cat: num - max number of unique values to recognize a column as categorical
:return
    "cat" if the column is categorical or "num" otherwise
'''
def utils_recognize_type(dtf, col, max_cat=20):
    if (dtf[col].dtype == "O") | (dtf[col].nunique() < max_cat):
        return "cat"
    else:
        return "num"


def ANOVA(y, x, dtf):
    cat, num = y, x
    model = smf.ols(num+' ~ '+cat, data=dtf).fit()
    table = sm.stats.anova_lm(model)
    p = table["PR(>F)"][0]
    coeff, p = None, round(p, 3)
    conclusion = "Correlated" if p < 0.05 else "Non-Correlated"
    print("Anova F: the variables " + y + " and " + x + " are", conclusion, "(p-value: "+str(p)+")")

def  Cramer(y, x, dtf):
    cont_table = pd.crosstab(index=dtf[x], columns=dtf[y])
    chi2_test = scipy.stats.chi2_contingency(cont_table)
    chi2, p = chi2_test[0], chi2_test[1]
    n = cont_table.sum().sum()
    phi2 = chi2/n
    r,k = cont_table.shape
    phi2corr = max(0, phi2-((k-1)*(r-1))/(n-1))
    rcorr = r-((r-1)**2)/(n-1)
    kcorr = k-((k-1)**2)/(n-1)
    coeff = np.sqrt(phi2corr/min((kcorr-1), (rcorr-1)))
    coeff, p = round(coeff, 3), round(p, 3)
    conclusion = "Significant" if p < 0.05 else "Non-Significant"
    print(y + ", " + x)
    print("Cramer Correlation:", coeff, conclusion, "(p-value:"+str(p)+")")



def main():
    print("hi")
    dtf = pd.read_csv(CSV_INPUT)
    print(dtf.head())

    # dic_cols = {col: utils_recognize_type(dtf, col, max_cat=20) for col in dtf.columns}
    # heatmap = dtf.isnull()
    # for k, v in dic_cols.items():
    #     if v == "num":
    #         heatmap[k] = heatmap[k].apply(lambda x: 0.5 if x is False else 1)
    #     else:
    #         heatmap[k] = heatmap[k].apply(lambda x: 0 if x is False else 1)
    #         sns.heatmap(heatmap, cbar=False).set_title('Dataset Overview')
    # plt.show()
    # print("\033[1;37;40m Categerocial ", "\033[1;30;41m Numeric ", "\033[1;30;47m NaN ")

    dtf = dtf.set_index("PATNO")
    dtf = dtf.rename(columns={"subgroup":"Y"})

    y = "Y"
    ax = dtf[y].value_counts().sort_values().plot(kind="barh")
    totals= []
    for i in ax.patches:
        totals.append(i.get_width())
    total = sum(totals)
    for i in ax.patches:
         ax.text(i.get_width()+.3, i.get_y()+.20,
         str(round((i.get_width()/total)*100, 2))+'%',
         fontsize=10, color='black')
    ax.grid(axis="x")
    plt.suptitle(y, fontsize=20)
    plt.savefig("Y_histogram.png")
    plt.close()

    x = "BMI"
    fig, ax = plt.subplots(nrows=1, ncols=2,  sharex=False, sharey=False)
    fig.suptitle(x, fontsize=20)
    ### distribution
    ax[0].title.set_text('distribution')
    variable = dtf[x].fillna(dtf[x].mean())
    breaks = np.quantile(variable, q=np.linspace(0, 1, 11))
    variable = variable[ (variable > breaks[0]) & (variable <
                        breaks[10]) ]
    sns.distplot(variable, hist=True, kde=True, kde_kws={"shade": True}, ax=ax[0])
    des = dtf[x].describe()
    ax[0].axvline(des["25%"], ls='--')
    ax[0].axvline(des["mean"], ls='--')
    ax[0].axvline(des["75%"], ls='--')
    ax[0].grid(True)
    des = round(des, 2).apply(lambda x: str(x))
    box = '\n'.join(("min: "+des["min"], "25%: "+des["25%"], "mean: "+des["mean"], "75%: "+des["75%"], "max: "+des["max"]))
    ax[0].text(0.95, 0.95, box, transform=ax[0].transAxes, fontsize=10, va='top', ha="right", bbox=dict(boxstyle='round', facecolor='white', alpha=1))### boxplot
    ax[1].title.set_text('outliers (log scale)')
    tmp_dtf = pd.DataFrame(dtf[x])
    tmp_dtf[x] = np.log(tmp_dtf[x])
    tmp_dtf.boxplot(column=x, ax=ax[1])
    plt.savefig("BMI_distribution.png")
    plt.close()


    # cat, num = "Y", "Age"
    # fig, ax = plt.subplots(nrows=1, ncols=3, sharex=False, sharey=False)
    # fig.suptitle(x + "   vs   " + y, fontsize=20)
    # ### distribution
    # ax[0].title.set_text('density')
    # for i in dtf[cat].unique():
    #     sns.distplot(dtf[dtf[cat] == i][num], hist=False, label=i, ax=ax[0])
    # ax[0].grid(True)  ### stacked
    # ax[1].title.set_text('bins')
    # breaks = np.quantile(dtf[num], q=np.linspace(0, 1, 11))
    # tmp = dtf.groupby([cat, pd.cut(dtf[num], breaks, duplicates='drop')]).size().unstack().T
    # tmp = tmp[dtf[cat].unique()]
    # tmp["tot"] = tmp.sum(axis=1)
    # for col in tmp.drop("tot", axis=1).columns:
    #     tmp[col] = tmp[col] / tmp["tot"]
    # tmp.drop("tot", axis=1).plot(kind='bar', stacked=True, ax=ax[1], legend=False, grid=True)  ### boxplot
    # ax[2].title.set_text('outliers')
    # sns.catplot(x=cat, y=num, data=dtf, kind="box", ax=ax[2])
    # ax[2].grid(True)
    # plt.show()

    # correlation with numerical variable
    ANOVA("Y", "BMI", dtf=dtf)

    # x, y = "Sex", "Y"
    # fig, ax = plt.subplots(nrows=1, ncols=2,  sharex=False, sharey=False)
    # fig.suptitle(x+"   vs   "+y, fontsize=20)### count
    # ax[0].title.set_text('count')
    # order = dtf.groupby(x)[y].count().index.tolist()
    # sns.catplot(x=x, hue=y, data=dtf, kind='count', order=order, ax=ax[0])
    # ax[0].grid(True)### percentage
    # ax[1].title.set_text('percentage')
    # a = dtf.groupby(x)[y].count().reset_index()
    # a = a.rename(columns={y:"tot"})
    # b = dtf.groupby([x,y])[y].count()
    # b = b.rename(columns={y:0}).reset_index()
    # b = b.merge(a, how="left")
    # b["%"] = b[0] / b["tot"] *100
    # sns.barplot(x=x, y="%", hue=y, data=b,
    #             ax=ax[1]).get_legend().remove()
    # ax[1].grid(True)
    # plt.show()

    # correlation with categorical variable
    Cramer(y, "sex", dtf=dtf)

    ## split data
    dtf_train, dtf_test = model_selection.train_test_split(dtf, test_size=0.3)
    ## print info

    print("X_train shape:", dtf_train.drop("Y",axis=1).shape, "| X_test shape:", dtf_test.drop("Y",axis=1).shape)
    print("y_train mean:", round(np.mean(dtf_train["Y"]),2), "| y_test mean:", round(np.mean(dtf_test["Y"]),2))
    print(dtf_train.shape[1], "features:", dtf_train.drop("Y",axis=1).columns.to_list())

    # normalize trainset
    d_train = dtf_train.drop("Y", axis=1)
    d_train = d_train.drop("NHY", axis=1)
    d_train = d_train.drop("age_at_visit", axis=1)
    print("features trainset:", d_train.columns.to_list())
    print(d_train.head())

    # fill empty value with mean
    for c in list(d_train):
        print(c)
        d_train[c] = d_train[c].fillna(d_train[c].mean())
    # dtf_train["Age"] = dtf_train["Age"].fillna(dtf_train["Age"].mean())
    print(d_train.head())

    # scale value
    scaler = preprocessing.MinMaxScaler(feature_range=(0,1))
    X = scaler.fit_transform(d_train)
    d_train_scaled = pd.DataFrame(X, columns=d_train.columns, index=dtf_train.index)
    #d_train_scaled["Y"] = dtf_train["Y"]
    print(d_train_scaled.head())

    # normalize testset
    d_test = dtf_test.drop("Y", axis=1)
    d_test = d_test.drop("NHY", axis=1)
    d_test = d_test.drop("age_at_visit", axis=1)
    print("features testset:", d_test.columns.to_list())
    print(d_test.head())

    # fill empty value with mean
    for c in list(d_test):
        d_test[c] = d_test[c].fillna(d_test[c].mean())
    # dtf_train["Age"] = dtf_train["Age"].fillna(dtf_train["Age"].mean())
    print(d_test.head())

    # scale value
    scaler = preprocessing.MinMaxScaler(feature_range=(0,1))
    X = scaler.fit_transform(d_test)
    d_test_scaled = pd.DataFrame(X, columns=d_test.columns, index=d_test.index)
    #d_test_scaled["Y"] = dtf_train["Y"]
    print(d_test_scaled.head())


    corr_matrix = dtf.copy()
    for col in corr_matrix.columns:
        if corr_matrix[col].dtype == "O":
             corr_matrix[col] = corr_matrix[col].factorize(sort=True)[0]
    corr_matrix = corr_matrix.corr(method="pearson")
    sns.heatmap(corr_matrix, vmin=-1., vmax=1., annot=True, fmt='.2f', cmap="YlGnBu", cbar=True, linewidths=0.5)
    plt.title("pearson correlation")
    plt.savefig("corr_matrix_train.png")
    plt.close()

    #####################################################
    # FEATURE SELECTION: ANOVA + Lasso
    #####################################################
    X = d_train_scaled.values

    y = dtf_train["Y"].values
    feature_names = d_train_scaled.columns  ## Anova
    selector = feature_selection.SelectKBest(score_func=
                                             feature_selection.f_classif, k='all').fit(X, y)
    anova_selected_features = feature_names[selector.get_support()]

    ## Lasso regularization
    selector = feature_selection.SelectFromModel(estimator=
                                                 linear_model.LogisticRegression(C=1, penalty="l1",
                                                                                 solver='liblinear'),
                                                 max_features=5).fit(X, y)
    lasso_selected_features = feature_names[selector.get_support()]

    ## Plot
    dtf_features = pd.DataFrame({"features": feature_names})
    dtf_features["anova"] = dtf_features["features"].apply(lambda x: "anova" if x in anova_selected_features else "")
    dtf_features["num1"] = dtf_features["features"].apply(lambda x: 1 if x in anova_selected_features else 0)
    dtf_features["lasso"] = dtf_features["features"].apply(lambda x: "lasso" if x in lasso_selected_features else "")
    dtf_features["num2"] = dtf_features["features"].apply(lambda x: 1 if x in lasso_selected_features else 0)
    dtf_features["method"] = dtf_features[["anova", "lasso"]].apply(lambda x: (x[0] + " " + x[1]).strip(), axis=1)
    dtf_features["selection"] = dtf_features["num1"] + dtf_features["num2"]
    sns.barplot(y="features", x="selection", hue="method", data=dtf_features.sort_values("selection", ascending=False),
                dodge=False)
    plt.savefig("feature_importance.png")
    plt.close()

    ############################################################
    # FEATURE SELECTION: RANDOM FOREST
    ############################################################
    X = d_train_scaled.values
    y = dtf_train["Y"].values
    feature_names = d_train_scaled.columns.tolist()
    ## Importance
    model = ensemble.RandomForestClassifier(n_estimators=100,
                                            criterion="entropy", random_state=0)
    model.fit(X, y)
    importances = model.feature_importances_
    ## Put in a pandas dtf
    dtf_importances = pd.DataFrame({"IMPORTANCE": importances,
                                    "VARIABLE": feature_names}).sort_values("IMPORTANCE",
                                                                            ascending=False)
    dtf_importances['cumsum'] = dtf_importances['IMPORTANCE'].cumsum(axis=0)
    dtf_importances = dtf_importances.set_index("VARIABLE")

    ## Plot
    fig, ax = plt.subplots(nrows=1, ncols=2, sharex=False, sharey=False)
    fig.suptitle("Features Importance", fontsize=20)
    ax[0].title.set_text('variables')
    dtf_importances[["IMPORTANCE"]].sort_values(by="IMPORTANCE").plot(
        kind="barh", legend=False, ax=ax[0]).grid(axis="x")
    ax[0].set(ylabel="")
    ax[1].title.set_text('cumulative')
    dtf_importances[["cumsum"]].plot(kind="line", linewidth=4,
                                     legend=False, ax=ax[1])
    ax[1].set(xlabel="", xticks=np.arange(len(dtf_importances)),
              xticklabels=dtf_importances.index)
    plt.xticks(rotation=70)
    plt.grid(axis='both')
    plt.savefig("feature_importance2.png")
    plt.close()

    ##############################################################
    X_names = ["sex", "educyrs", "race", "fampd", "handed", "BMI", "agediag", "ageonset", "duration", "upsitpctl",
               "MOCA", "bjlot", "hvlt_discrimination"]
    X_train = d_train_scaled[X_names].values
    y_train = dtf_train["Y"].values
    X_test = d_test_scaled[X_names].values
    y_test = dtf_test["Y"].values
    print(X_test.shape)
    print(y_test.shape)

    ##############################################################
    # GradientBoostingClassifier
    ##############################################################
    ## call model
    begin = datetime.datetime.now().replace(microsecond=0)
    print("Start search: {}\n".format(begin), file=open(FILE_LOG, 'a'), flush=True)
    model = ensemble.GradientBoostingClassifier()
    ## define hyperparameters combinations to try
    param_dic = {'learning_rate':[0.15,0.1,0.05,0.01,0.005,0.001],      #weighting factor for the corrections by new trees when added to the model
    'n_estimators':[100,250,500,750,1000,1250,1500,1750],  #number of trees added to the model
    'max_depth':[2,3,4,5,6,7],    #maximum depth of the tree
    'min_samples_split':[2,4,6,8,10,20,40,60,100],    #sets the minimum number of samples to split
    'min_samples_leaf':[1,3,5,7,9],     #the minimum number of samples to form a leaf
    'max_features':[2,3,4,5,6,7],     #square root of features is usually a good starting point
    'subsample':[0.7,0.75,0.8,0.85,0.9,0.95,1]}       #the fraction of samples to be used for fitting the individual base learners. Values lower than 1 generally lead to a reduction of variance and an increase in bias.## random search
    random_search = model_selection.RandomizedSearchCV(model,
           param_distributions=param_dic, n_iter=1000,
           scoring="accuracy").fit(X_train, y_train)
    end = datetime.datetime.now().replace(microsecond=0)
    search_time = end - begin
    print("Search time: {}\n".format(search_time), file=open(FILE_LOG, 'a'), flush=True)
    print("Best Model parameters:", random_search.best_params_, file=open(FILE_LOG, 'a'), flush=True)
    print("Best Model mean accuracy:", random_search.best_score_, file=open(FILE_LOG, 'a'), flush=True)
    model = random_search.best_estimator_

    ##########################################################
    # ROC and AUC
    ##########################################################
    cv = model_selection.StratifiedKFold(n_splits=10, shuffle=True)
    tprs, aucs = [], []
    mean_fpr = np.linspace(0, 1, 100)
    fig = plt.figure()
    i = 1
    for train, test in cv.split(X_train, y_train):
        prediction = model.fit(X_train[train],
                               y_train[train]).predict_proba(X_train[test])
        fpr, tpr, t = metrics.roc_curve(y_train[test], prediction[:, 1])
        tprs.append(np.interp(mean_fpr, fpr, tpr))
        roc_auc = metrics.auc(fpr, tpr)
        aucs.append(roc_auc)
        plt.plot(fpr, tpr, lw=2, alpha=0.3, label='ROC fold %d (AUC = %0.2f)' % (i, roc_auc))
        i = i + 1

    plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='black')
    mean_tpr = np.mean(tprs, axis=0)
    mean_auc = metrics.auc(mean_fpr, mean_tpr)
    plt.plot(mean_fpr, mean_tpr, color='blue', label=r'Mean ROC (AUC = %0.2f )' % (mean_auc), lw=2, alpha=1)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('K-Fold Validation')
    plt.legend(loc="lower right")
    plt.savefig("ROC.png")
    plt.close()


    ##########################################################
    # TRAIN and TEST
    ##########################################################
    ## train
    begin = datetime.datetime.now().replace(microsecond=0)
    print("Start training: {}\n".format(begin), file=open(FILE_LOG, 'a'), flush=True)
    model.fit(X_train, y_train)
    end = datetime.datetime.now().replace(microsecond=0)
    print("Training time: {}\n".format(end - begin), file=open(FILE_LOG, 'a'), flush=True)
    ## test
    predicted_prob = model.predict_proba(X_test)[:,1]
    predicted = model.predict(X_test)

    ####################################################
    # ACCURACY and AUC on test set
    ####################################################
    ## Accuray e AUC
    accuracy = metrics.accuracy_score(y_test, predicted)
    auc = metrics.roc_auc_score(y_test, predicted_prob)
    print("Accuracy on testset (overall correct predictions):", round(accuracy, 2), file=open(FILE_LOG, 'a'), flush=True)
    print("Auc on testset:", round(auc, 2), file=open(FILE_LOG, 'a'), flush=True)

    ## Precision e Recall
    recall = metrics.recall_score(y_test, predicted)
    precision = metrics.precision_score(y_test, predicted)
    print("Recall (all 1s predicted right):", round(recall, 2))
    print("Precision (confidence when predicting a 1):", round(precision, 2))
    print("Detail:", file=open(FILE_LOG, 'a'), flush=True)
    print(metrics.classification_report(y_test, predicted, target_names=[str(i) for i in np.unique(y_test)]),
          file=open(FILE_LOG, 'a'), flush=True)

    ####################################################
    # CONFUSION MATRIX on testset
    ####################################################
    classes = np.unique(y_test)
    fig, ax = plt.subplots()
    cm = metrics.confusion_matrix(y_test, predicted, labels=classes)
    sns.heatmap(cm, annot=True, fmt='d', cmap=plt.cm.Blues, cbar=False)
    ax.set(xlabel="Pred", ylabel="True", title="Confusion matrix")
    ax.set_yticklabels(labels=classes, rotation=0)
    plt.savefig("conf_matrix_test.png")
    plt.close()

    ####################################################
    # ROC and P-R curves on testset
    ####################################################
    classes = np.unique(y_test)
    fig, ax = plt.subplots(nrows=1, ncols=2)
    ## plot ROC curve
    fpr, tpr, thresholds = metrics.roc_curve(y_test, predicted_prob)
    roc_auc = metrics.auc(fpr, tpr)
    ax[0].plot(fpr, tpr, color='darkorange', lw=3, label='area = %0.2f' % roc_auc)
    ax[0].plot([0, 1], [0, 1], color='navy', lw=3, linestyle='--')
    ax[0].hlines(y=recall, xmin=0, xmax=1 - cm[0, 0] / (cm[0, 0] + cm[0, 1]), color='red', linestyle='--', alpha=0.7,
                 label="chosen threshold")
    ax[0].vlines(x=1 - cm[0, 0] / (cm[0, 0] + cm[0, 1]), ymin=0, ymax=recall, color='red', linestyle='--', alpha=0.7)
    ax[0].set(xlabel='False Positive Rate', ylabel="True Positive Rate (Recall)",
              title="Receiver operating characteristic")
    ax[0].legend(loc="lower right")
    ax[0].grid(True)
    ## annotate ROC thresholds
    thres_in_plot = []
    for i, t in enumerate(thresholds):
        t = np.round(t, 1)
        if t not in thres_in_plot:
            ax[0].annotate(t, xy=(fpr[i], tpr[i]), xytext=(fpr[i], tpr[i]),
                        textcoords='offset points', ha='left', va='bottom')
            thres_in_plot.append(t)
        else:
            next
    ## plot P-R curve
    precisions, recalls, thresholds = metrics.precision_recall_curve(y_test, predicted_prob)
    roc_auc = metrics.auc(recalls, precisions)
    ax[1].plot(recalls, precisions, color='darkorange', lw=3, label='area = %0.2f' % roc_auc)
    ax[1].plot([0, 1], [(cm[1, 0] + cm[1, 0]) / len(y_test), (cm[1, 0] + cm[1, 0]) / len(y_test)], linestyle='--',
               color='navy', lw=3)
    ax[1].hlines(y=precision, xmin=0, xmax=recall, color='red', linestyle='--', alpha=0.7, label="chosen threshold")
    ax[1].vlines(x=recall, ymin=0, ymax=precision, color='red', linestyle='--', alpha=0.7)
    ax[1].set(xlabel='Recall', ylabel="Precision", title="Precision-Recall curve")
    ax[1].legend(loc="lower left")
    ax[1].grid(True)  ## annotate P-R thresholds
    thres_in_plot = []
    for i, t in enumerate(thresholds):
        t = np.round(t, 1)
        if t not in thres_in_plot:
            ax[1].annotate(np.round(t, 1), xy=(recalls[i], precisions[i]),
                        xytext=(recalls[i], precisions[i]),
                        textcoords='offset points', ha='left', va='bottom')
            thres_in_plot.append(t)
        else:
            next
    plt.savefig("ROC_PR_testset.png")
    plt.close()

    # ####################################################
    # # THRESHOLD SELECTION
    # ####################################################
    # ## calculate scores for different thresholds
    # dic_scores = {'accuracy': [], 'precision': [], 'recall': [], 'f1': []}
    # XX_train, XX_test, yy_train, yy_test = model_selection.train_test_split(X_train, y_train, test_size=0.2)
    # predicted_prob = model.fit(XX_train, yy_train).predict_proba(XX_test)[:, 1]
    # thresholds = []
    # for threshold in np.arange(0.1, 1, step=0.1):
    #     predicted = (predicted_prob > threshold)
    #     thresholds.append(threshold)
    #     dic_scores["accuracy"].append(metrics.accuracy_score(yy_test, predicted))
    # dic_scores["precision"].append(metrics.precision_score(yy_test, predicted))
    # dic_scores["recall"].append(metrics.recall_score(yy_test, predicted))
    # dic_scores["f1"].append(metrics.f1_score(yy_test, predicted))
    #
    # ## plot
    # dtf_scores = pd.DataFrame(dic_scores).set_index(pd.Index(thresholds))
    # dtf_scores.plot(ax=ax, title="Threshold Selection")
    # plt.savefig("threshold_sel.png")

    ############################################
    # Explainability
    ############################################
    print("True:", y_test[4], "--> Pred:", predicted[4], "| Prob:", np.max(predicted_prob[4]), file=open(FILE_LOG, 'a'), flush=True)
    explainer = lime_tabular.LimeTabularExplainer(training_data=X_train, feature_names=X_names, class_names=np.unique(y_train), mode="classification")
    explained = explainer.explain_instance(X_test[4], model.predict_proba, num_features=10)
    print("Explain prediction: ", file=open(FILE_LOG, 'a'), flush=True)
    print(explained.as_list(), file=open(FILE_LOG, 'a'), flush=True)
    explained.as_pyplot_figure()
    plt.savefig("explain.png")
    plt.close()

    # ######################################################
    # # classification regions
    # ######################################################
    # ## PCA
    # pca = decomposition.PCA(n_components=2)
    # X_train_2d = pca.fit_transform(X_train)
    # X_test_2d = pca.transform(X_test)  ## train 2d model
    # model_2d = ensemble.GradientBoostingClassifier()
    # model_2d.fit(X_train, y_train)
    #
    # ## plot classification regions
    # from matplotlib.colors import ListedColormap
    #
    # colors = {np.unique(y_test)[0]: "black", np.unique(y_test)[1]: "green"}
    # X1, X2 = np.meshgrid(np.arange(start=X_test[:, 0].min() - 1, stop=X_test[:, 0].max() + 1, step=0.01),
    #                      np.arange(start=X_test[:, 1].min() - 1, stop=X_test[:, 1].max() + 1, step=0.01))
    # fig, ax = plt.subplots()
    # Y = model_2d.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape)
    # ax.contourf(X1, X2, Y, alpha=0.5, cmap=ListedColormap(list(colors.values())))
    # ax.set(xlim=[X1.min(), X1.max()], ylim=[X2.min(), X2.max()], title="Classification regions")
    # for i in np.unique(y_test):
    #     ax.scatter(X_test[y_test == i, 0], X_test[y_test == i, 1],
    #                c=colors[i], label="true " + str(i))
    # plt.legend()
    # plt.savefig("classif_regions.png")
    # plt.close()


if __name__ == '__main__':
    main()