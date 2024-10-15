import feyn
import numpy as np
import pandas as pd
import sklearn.model_selection

#Load the dataset
data = r"C:\Users\user\Desktop\archive\data.csv"
df = pd.read_csv(data)
df.iloc[:,1400:1500]

#Target balance
df["vital.status"].value_counts()

train, test = sklearn.model_selection.train_test_split(df,stratify=df["vital.status"], train_size=.66, random_state=42)

#Allocate a QLattice
ql = feyn.connect_qlattice()

#Search for the best model
ql.reset(random_seed=42)
models = ql.auto_run(train, output_name="vital.status", kind="classification", criterion = None, n_epochs=25, max_complexity=3)

models[0].plot(train, test)

# Training Data
models[1].plot_response_2d(train)

#Test Data
models[1].plot_response_2d(test)

#Test Data
models[1].plot_response_2d(test)

rf = feyn.reference.RandomForestClassifier(train, output_name="vital.status", random_state = 42)
gb = feyn.reference.GradientBoostingClassifier(train, output_name="vital.status", random_state = 42)
lr = feyn.reference.LogisticRegressionClassifier(train, output_name="vital.status", max_iter=10000, random_state = 42)

rf.plot_roc_curve(test, label="Random Forest")
gb.plot_roc_curve(test, label="Gradient Boosting")
lr.plot_roc_curve(test, label="Logistic Regression")

#The Multi-Omics Capabilities of the QLattice
def feature_correlations_map(train, target, correlation='pearson', scale_features=True, abs_values = False):

    from sklearn.decomposition import PCA
    import plotly.express as px
    
    target_abs = f"{target}_abs"

    data_corr = train.corr(correlation).drop(target, axis = 0)

    target_corr = data_corr[target].values.copy()
    data_corr = data_corr.drop(target, axis = 1)

    features = data_corr.columns

    if scale_features:
        from sklearn.preprocessing import StandardScaler
        x = data_corr.loc[:, features].values
        data_corr = StandardScaler().fit_transform(x)
        from sklearn.decomposition import PCA
        pca = PCA(n_components=2)
    principalComponents = pca.fit_transform(data_corr)
    principalDf = pd.DataFrame(data = principalComponents, columns = ['PC1', 'PC2'])

    principalDf[target] = target_corr
    principalDf[target_abs] = np.abs(target_corr)
    principalDf['Features'] = features
    
    color_feature = target
    
    if abs_values:
        color_feature = target

    fig = px.scatter(principalDf, x="PC1",
                    y="PC2", color=color_feature, hover_name="Features", width=1000, height=800)
    fig.add_annotation(text=f'Variance explained <br>PC1: {"{:.0%}".format(pca.explained_variance_ratio_[0])} <br>PC2: {"{:.0%}".format(pca.explained_variance_ratio_[1])}', 
                    align='left',
                    showarrow=False,
                    xref='paper',
                    yref='paper',
                    x=0.98,
                    y=0.98,
                    bordercolor='black',
                    borderwidth=1)

    fig.show()

    multi_model = feyn.Model.load(r"C:\Users\user\Desktop\archive\model1_200its_mc7.model")
    print("Model Loaded:",multi_model)
    multi_model.plot(train, test)
    multi_model.plot_response_2d(df, fixed = {"mu_TNXB" : 0, "cn_ANKRD30B" : -1})
    multi_model.plot_response_2d(df, fixed = {"mu_TNXB" : 1, "cn_ANKRD30B" : -1})



