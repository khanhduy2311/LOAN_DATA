import pandas as pd
import matplotlib.pyplot as plt
import sklearn as sk
import scipy.cluster.hierarchy as sch
import seaborn as sns
from sklearn.preprocessing import StandardScaler

# Đọc dữ liệu
loan_data = pd.read_csv("loan_data.csv")
loan_data.head()

# Tính phần trăm giá trị bị thiếu
percent_missing = round(100 * (loan_data.isnull().sum())/len(loan_data), 2)
print(percent_missing)

# Loại bỏ các cột không cần thiết
clean_data = loan_data.drop(['purpose', 'not.fully.paid'], axis = 1)
clean_data.info()

def show_box(df):
    plt.rcParams['figure.figsize'] = [14,6]
    sns.boxplot(data = df, orient = "v")
    plt.title("Outliers Distribution", fontsize = 16)
    plt.ylabel("Range", fontweight = 'bold')
    plt.xlabel("Attributes", fontweight = 'bold')
    plt.show()

def remove_outliers(data):
    df = data.copy()
    for col in list(df.columns):
        Q1 = df[str(col)].quantile(0.05)
        Q3 = df[str(col)].quantile(0.95)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5*IQR
        upper_bound = Q3 + 1.5*IQR
        df = df[(df[str(col)] >= lower_bound) & (df[str(col)] <= upper_bound)]
    return df

# Loại bỏ outliers
without_outliers = remove_outliers(clean_data)

# Scale dữ liệu
data_scaler = StandardScaler()
scaled_data = data_scaler.fit_transform(without_outliers)

# Phân cụm
complete_clustering = sch.linkage(scaled_data, method = "complete", metric = "euclidean")
average_clustering = sch.linkage(scaled_data, method = "average", metric = "euclidean")
single_clustering = sch.linkage(scaled_data, method = "single", metric = "euclidean")
#dendrogram(complete_clustering)
#dendrogram(average_clustering)
#dendrogram(single_clustering)

# Cắt cụm
cluster_labels = sch.cut_tree(average_clustering, n_clusters=2).reshape(-1)
without_outliers["Cluster"] = cluster_labels

# Vẽ boxplot
sns.boxplot(x='Cluster', y='fico', data=without_outliers)
plt.show()
