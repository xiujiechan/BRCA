import numpy as np
import pandas as pd
import os
from sklearn.linear_model import LinearRegression
from scipy.stats import ttest_ind
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

#Section 1:Data Exploration
# Open the dataset
df = pd.read_csv(r"C:\Users\user\Desktop\archive\brca_data_w_subtypes.csv")

# View the first 5 rows
print(df.head())

print("Presenting the DataFrame:")
print("")
print("Variables: Four types of omics.")
print("Number of RNAseq variables:",
        len([match for match in df.columns if match.startswith("rs")]))
print("Number of Copy Number variables:",
        len([match for match in df.columns if match.startswith("cn")]))
print("Number of Mutation variables:",
        len([match for match in df.columns if match.startswith("mu")]))
print("Number of Protein variables:",
        len([match for match in df.columns if match.startswith("pp")]))

print("")
print("Outcomes: Five Outcomes.")
print("Vital Status:",df["vital.status"].unique())
print("Progesterone Receptors:",(df["PR.Status"]).unique())
print("Estrogen Receptors:",(df["ER.Status"]).unique())
print("HER2 Status",(df["HER2.Final.Status"]).unique())
print("Histological Cancer Subtype",(df["histological.type"]).unique())


#Section 2:Single-gene study
# Isolate columns with information about this gene
df_myh11 = df[[col for col in df.columns if "MYH11" in col]]
print("MYH11 Variables:",list(df_myh11.columns))

for col in df_myh11.columns:

    # perform the t-test
    print("T-Test on relationship between %s and patient survival"%col)
    print("Mean in Dead:",df.loc[df["vital.status"]==0,col].mean())
    print("Mean in Alive:",df.loc[df["vital.status"]==1,col].mean())
    t_stat, p_val = ttest_ind(df.loc[df["vital.status"]==0,col],
                                df.loc[df["vital.status"]==1,col])

    # print the results
    print('T-statistic: ', t_stat)
    print('P-value: ', p_val)
    print("")

    # calculate effect of CN on status
model = LinearRegression(fit_intercept=True)
model.fit(df[["cn_MYH11"]], df["vital.status"])
cn_on_status = model.coef_[0]

# calculate effect of CN on RNASeq
model = LinearRegression(fit_intercept=True)
model.fit(df[["cn_MYH11"]], df["rs_MYH11"])
cn_on_rnaseq = model.coef_[0]

# calculate effect of RNASeq on status controlling for CN
model = LinearRegression(fit_intercept=True)
model.fit(df[["cn_MYH11","rs_MYH11"]], df["vital.status"])
cn_on_status_controlled = model.coef_[0]
rs_on_status_controlled = model.coef_[1]

print("Total Effect of CN on Status:",cn_on_status)
print("Direct Effect of CN on Status:",cn_on_status_controlled)
print("Mediated Effect of CN by RNAseq on Status:",cn_on_rnaseq*rs_on_status_controlled)

#Section 3:Clustering using Multi-Omics
outcomes = df[["vital.status","PR.Status","ER.Status","HER2.Final.Status","histological.type"]]
df_mu = df[[col for col in df if col.startswith('mu')]]
df_cn = df[[col for col in df if col.startswith('cn')]]
df_rs = df[[col for col in df if col.startswith('rs')]]
df_pp = df[[col for col in df if col.startswith('pp')]]

# Calculate Pearson's correlation coefficients between copy number and RNA-seq data
correlation_matrix = np.corrcoef(df_cn, df_rs, rowvar=False)[:df_cn.shape[1], df_cn.shape[1]:]

# Create a dataframe from the correlation matrix
correlation_df = pd.DataFrame(correlation_matrix, index=df_cn.columns, columns=df_rs.columns)

# Plot the heatmap
plt.figure(figsize=(20, 20))
sns.clustermap(correlation_df, cmap="coolwarm")

# Customize plot appearance
plt.title("Pairwise Pearson's Correlations between Copy Number and RNA-seq Data")

# Show the plot
plt.show()

# Plot the heatmap
plt.figure(figsize=(20, 20))
sns.clustermap(correlation_df, cmap="coolwarm")

# Customize plot appearance
plt.title("Pairwise Pearson's Correlations between Copy Number and RNA-seq Data")

# Show the plot
plt.show()
plt.ylabel("Principal Component 2")

# Show the plot
plt.show()

from sklearn.cluster import KMeans
# Standardize the data
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
standardized_data = scaler.fit_transform(df.drop(columns=outcomes.columns))
# Check if standardized_data is correctly standardized
print(standardized_data[:5])

# Apply PCA to reduce dimensionality
n_components = 2
pca = PCA(n_components=n_components)
reduced_data = pca.fit_transform(standardized_data)

# Apply KMeans clustering
n_clusters = 4
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
cluster_assignments = kmeans.fit_predict(standardized_data)

# Generate set of colors for clusters
colors = []
for ii in range(n_clusters):
    colors = [f"#{np.random.randint(0, 0xFFFFFF):06x}" for _ in range(n_clusters)]

    #colors.append("#{:06x}".format(np.random.randint(0, 0xFFFFFF)))


# Plot the reduced data with cluster colors
plt.figure(figsize=(10, 10))
for cluster in range(n_clusters):
    cluster_points = reduced_data[cluster_assignments == cluster]
    plt.scatter(cluster_points[:, 0], cluster_points[:, 1], s=50, label=f"Cluster {cluster + 1}",color=colors[cluster])

# Customize plot appearance
plt.title("PCA of Combined Copy Number and RNA-seq Data with KMeans Clustering")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.legend()

# Show the plot
plt.show()

# Let's overlay survival rate on top of these clusters?

survived = (outcomes["vital.status"]).to_numpy()


# Define marker shapes for the survived array
marker_shapes = {0: '.', 1: 'o'}

# Plot the reduced data with cluster colors and different shapes for survived values
plt.figure(figsize=(10, 10))
for cluster in range(n_clusters):
    color = colors[cluster]
    for survived_value, marker_shape in marker_shapes.items():
        cluster_points = reduced_data[(cluster_assignments == cluster) & (survived == survived_value)]
        plt.scatter(cluster_points[:, 0], cluster_points[:, 1], s=50, marker=marker_shape,
                    label=f"Cluster {cluster + 1}, Survived: {survived_value}",color=color)

# Customize plot appearance
plt.title("PCA of Combined Copy Number and RNA-seq Data with KMeans Clustering")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.legend()

# Show the plot
plt.show()

# Proportions of survival in each cluster
cluster_survival = []
for cluster in range(n_clusters):
    cluster_survival.append(np.sum((cluster_assignments==cluster)&(survived==1))/np.sum(cluster_assignments==cluster))

labels = []
for ii in range(n_clusters):
    labels.append("Cluster %i"%(ii+1))

# create a bar plot
plt.bar(labels, cluster_survival,color=colors)
# plot mean
plt.plot([-1,4],[np.mean(survived),np.mean(survived)],'--',color='gray')

# add a title to the plot
plt.title('Proportion of survived members in each cluster')

# add labels to the x and y axes
plt.xlabel('X-axis label')
plt.ylabel('Y-axis label')
# display the plot
plt.show()

# Let's overlay cancer type on top of these clusters?
cancer_types_desc = ["infiltrating ductal carcinoma","infiltrating lobular carcinoma"]
cancer = (outcomes["histological.type"]).to_numpy()


# Define marker shapes for the cancer array
marker_shapes = {"infiltrating ductal carcinoma": '.', "infiltrating lobular carcinoma": 'o'}

# Plot the reduced data with cluster colors and different shapes for cancer values
plt.figure(figsize=(10, 10))
for cluster in range(n_clusters):
    color = colors[cluster]
    for cancer_value, marker_shape in marker_shapes.items():
        cluster_points = reduced_data[(cluster_assignments == cluster) & (cancer == cancer_value)]
        plt.scatter(cluster_points[:, 0], cluster_points[:, 1], s=50, marker=marker_shape,
                    label=f"Cluster {cluster + 1}, Cancer Type: {cancer_value}",color=color)
        
        # Customize plot appearance
plt.title("PCA of Combined Copy Number and RNA-seq Data with KMeans Clustering")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.legend()

# Show the plot
plt.show()

# Proportions of infiltrating lobular carcinoma in each cluster
cluster_survival = []
for cluster in range(n_clusters):
    cluster_survival.append(np.sum((cluster_assignments==cluster)&(cancer=="infiltrating lobular carcinoma"))/np.sum(cluster_assignments==cluster))
    
labels = []
for ii in range(n_clusters):
    labels.append("Cluster %i"%(ii+1))

# create a bar plot
plt.bar(labels, cluster_survival,color=colors)
# plot mean
plt.plot([-1,4],[np.mean(cancer=="infiltrating lobular carcinoma"),
                    np.mean(cancer=="infiltrating lobular carcinoma")],'--',color='gray')

# add a title to the plot
plt.title('Proportion of infiltrating lobular carcinoma in each cluster')

# add labels to the x and y axes
plt.xlabel('X-axis label')
plt.ylabel('Y-axis label')

# display the plot
plt.show()
