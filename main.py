import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris, load_breast_cancer, load_wine
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA

# Define a function to load datasets and cache the results
@st.cache_data
def load_data(name):
    if name == "Iris":
        data = load_iris()
        target_names = {0: 'Iris-setosa', 1: 'Iris-versicolor', 2: 'Iris-virginica'}
    elif name == "Breast Cancer":
        data = load_breast_cancer()
        target_names = {0: 'Malignant', 1: 'Benign'}
    else:
        data = load_wine()
        target_names = {0: 'Class 0', 1: 'Class 1', 2: 'Class 2'}
    return data, target_names

st.set_page_config(page_title="ClasPy ML App", page_icon="🔍")

st.markdown("""
    <div style="text-align: center;">
        <h1>ClasPy ML App</h1>
        <p>🔍 Explore classifiers to see which performs best on different datasets</p>
    </div>
    """, unsafe_allow_html=True)

# Add a divider
st.markdown("---")

dataset_name = st.sidebar.selectbox("Select Dataset", ("Iris", "Breast Cancer", "Wine")) 
classifier_name = st.sidebar.selectbox("Select Classifier", ("KNN", "SVM", "Random Forest")) 

# Load the selected dataset
data, target_names = load_data(dataset_name)

st.write(f"### Analysis of the {dataset_name} Dataset")

# Create a DataFrame
df = pd.DataFrame(data.data, columns=data.feature_names)
df['target'] = data.target

# Split the dataset into features and target variable
X = df.drop(columns=['target'])
y = df['target']

st.write(df.head(20))
st.write(f"Shape of dataset {X.shape} with {y.nunique()} classes: {list(target_names.values())}")

# Apply feature scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Add hyperparameter widgets based on the selected classifier
if classifier_name == "KNN":
    n_neighbors = st.sidebar.slider("Number of Neighbors (K)", 1, 15)
    weights = st.sidebar.selectbox("Weights", ("uniform", "distance"))
    algorithm = st.sidebar.selectbox("Algorithm", ("auto", "ball_tree", "kd_tree", "brute"))
    classifier = KNeighborsClassifier(n_neighbors=n_neighbors, weights=weights, algorithm=algorithm)
elif classifier_name == "SVM":
    C = st.sidebar.slider("C (Regularization parameter)", 0.01, 10.0)
    kernel = st.sidebar.selectbox("Kernel", ("linear", "poly", "rbf", "sigmoid"))
    classifier = SVC(C=C, kernel=kernel)
else:
    max_depth = st.sidebar.slider("Max Depth", 2, 15)
    n_estimators = st.sidebar.slider("Number of Estimators", 1, 100)
    min_samples_split = st.sidebar.slider("Min Samples Split", 2, 10)
    min_samples_leaf = st.sidebar.slider("Min Samples Leaf", 1, 10)
    classifier = RandomForestClassifier(
        max_depth=max_depth, 
        n_estimators=n_estimators, 
        min_samples_split=min_samples_split, 
        min_samples_leaf=min_samples_leaf
    )

# Split the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Train the classifier
classifier.fit(X_train, y_train)

# Make predictions
y_pred = classifier.predict(X_test)

# Calculate and display the accuracy
accuracy = accuracy_score(y_test, y_pred)
st.markdown(f"""
    <div style="font-size: 24px; font-weight: bold;">
        Accuracy: {accuracy:.2f}
    </div>
    """, unsafe_allow_html=True)

# Add a divider
st.markdown("---")

# Plotting section with PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# Create a DataFrame for the PCA results
df_pca = pd.DataFrame(X_pca, columns=['PCA1', 'PCA2'])
df_pca['target'] = y

# Plotting the PCA results with custom legend
st.write("### Data Visualization")
fig, ax = plt.subplots()
scatter = ax.scatter(df_pca['PCA1'], df_pca['PCA2'], c=df_pca['target'], cmap='viridis')
handles, _ = scatter.legend_elements()
legend1 = ax.legend(handles, list(target_names.values()), title="Classes")
ax.add_artist(legend1)

ax.set_xlabel('PCA1')
ax.set_ylabel('PCA2')
ax.set_title('PCA plot')
st.pyplot(fig)

# Add interpretation based on the selected dataset
st.markdown("""
1. **Principal Components**: 
    - `PCA1` and `PCA2` are the two principal components. They are linear combinations of the original features 
            that capture the most variance in the data.
    - `PCA1` captures the highest variance, while `PCA2` captures the second highest.
""")
if dataset_name == "Iris":
    st.markdown("""

    2. **Clusters**:
        - The plot shows three distinct clusters of data points, each representing a different class in the Iris dataset.
        - Each color represents a different class (Iris-setosa, Iris-versicolor, and Iris-virginica).

    3. **Separation of Classes**:
        - Iris-setosa is well-separated from Iris-versicolor (cyan) and Iris-virginica.
        - Iris-versicolor and Iris-virginica have some overlap but are still reasonably distinguishable.

    4. **Interpretation**:
        - **Iris-setosa**: The distinct separation of the purple cluster indicates that the features 
                for this class are significantly different from those of the other classes.
        - **Iris-versicolor and Iris-virginica**: The cyan and yellow clusters are closer together, indicating 
                that these classes have more similar features. The overlap suggests 
                that the model might find it more challenging to distinguish between these two classes compared to Iris-setosa.
    """)

elif dataset_name == "Breast Cancer":
    st.markdown("""

    2. **Clusters**:
        - The plot shows two distinct clusters of data points, representing the two classes in the Breast Cancer dataset: 
            Malignant and Benign.
        - Each color represents a different class: purple for Malignant and yellow for Benign.

    3. **Separation of Classes**:
        - There is a noticeable separation between the Malignant (purple) and Benign (yellow) classes.
        - The Malignant class (purple) is more dispersed, indicating greater variability in the data points.
        - The Benign class (yellow) is more tightly clustered, suggesting less variability among those data points.

    4. **Interpretation**:
        - **Malignant (Purple)**: The spread of the purple cluster indicates that the features of the Malignant class 
                have significant variability. This could mean that the malignant cases have diverse characteristics.
        - **Benign (Yellow)**: The tight clustering of the yellow points indicates that the Benign class 
                has less variability and more consistent/uniform features.
    """)
else:
    st.markdown("""

    2. **Clusters**:
        - The plot shows three distinct clusters of data points, representing the three classes in the Wine dataset: 
                Class 0, Class 1, and Class 2.
        - Each color represents a different class: purple for Class 0, cyan for Class 1, and yellow for Class 2.

    3. **Separation of Classes**:
        - Class 0 (purple) is fairly well-separated from Classes 1 (cyan) and 2 (yellow).
        - Classes 1 and 2 have some overlap, indicating that these two classes have more similar features compared to Class 0.

    4. **Interpretation**:
        - **Class 0 (Purple)**: The separation of the purple cluster indicates that the features 
                for Class 0 are distinct from those of the other classes.
        - **Classes 1 (Cyan) and 2 (Yellow)**: The overlap between the cyan and yellow clusters suggests that these classes 
                have similar features, making them harder to distinguish.
    """)

# Add a divider
st.markdown("---")

# Display explained variance and explained variance ratio
st.write("### Explained Variance and Explained Variance Ratio")
st.write(f"Explained Variance: {pca.explained_variance_}")
st.write(f"Explained Variance Ratio: {pca.explained_variance_ratio_}")

# Get the values from the calculated PCA explained variance
pca1_var = pca.explained_variance_[0]
pca2_var = pca.explained_variance_[1]
pca1_ratio = pca.explained_variance_ratio_[0] * 100  # Convert to percentage
pca2_ratio = pca.explained_variance_ratio_[1] * 100  # Convert to percentage

# Define the interpretation template
interpretation_template = """
1. **Explained Variance**:
    - The first principal component (PCA1) captures a variance of approximately `{pca1_var}`.
    - The second principal component (PCA2) captures a variance of approximately `{pca2_var}`.

2. **Explained Variance Ratio**:
    - The first principal component (PCA1) captures about `{pca1_ratio:.2f}%` of the total variance.
    - The second principal component (PCA2) captures about `{pca2_ratio:.2f}%` of the total variance.

3. **Key Points**:
    - **Dominance of PCA1**: The first principal component (PCA1) captures a significantly larger portion of the variance `({pca1_ratio:.2f}%)` 
      compared to the second principal component `({pca2_ratio:.2f}%)`. This indicates that most of the information in the dataset 
      can be explained by the first principal component alone.
    - **Dimensionality Reduction**: The high explained variance ratio of the first principal component suggests that 
      the data can be effectively reduced to one dimension without losing much information. 
      However, the second principal component still captures some additional variance, 
      which might be useful for further distinguishing between classes.
"""

# Render the interpretation
st.markdown(interpretation_template.format(pca1_var=pca1_var, pca2_var=pca2_var, pca1_ratio=pca1_ratio, pca2_ratio=pca2_ratio))
