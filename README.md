
# **PRODIGY\_INFOTECH\_DATASCIENCE: Task-03**

##  **Objective:**

The goal of this task is to build a **Decision Tree Classifier** to predict whether a customer will subscribe to a term deposit based on their **demographic and behavioral attributes**. We utilize the **Bank Marketing Dataset** from the **UCI Machine Learning Repository**, which contains rich information about marketing campaigns run by a Portuguese banking institution.

This task focuses on supervised learning techniques and helps in understanding model building, preprocessing, evaluation, and decision tree visualization.

---

##  **Dataset Used:**

* **Source:** UCI Machine Learning Repository
* **Dataset:** Bank Marketing Dataset
* **Link:** [https://archive.ics.uci.edu/ml/datasets/Bank+Marketing](https://archive.ics.uci.edu/ml/datasets/Bank+Marketing)
* **Format:** CSV (`bank.csv`)
* **Note:** The CSV can be directly loaded using its [data link](https://archive.ics.uci.edu/ml/machine-learning-databases/00222/bank.csv)

---

## 🛠 **Technologies Used:**

* **Python 3.11+**
* **Pandas** for data manipulation
* **NumPy** for numeric operations
* **Scikit-learn** for model building and evaluation
* **Matplotlib** and **Seaborn** for visualizations
* **Jupyter Notebook** or any Python IDE

---

##  **Working:**

###  **Data Loading:**

* Loaded the dataset directly from the UCI repository using `pandas.read_csv()`
* Used `.info()`, `.describe()`, and `.value_counts()` for basic EDA and understanding the class distribution

###  **Data Preprocessing:**

* Applied **one-hot encoding** to convert categorical variables to numeric using `pd.get_dummies()`
* Target column `y` was converted into binary values (yes = 1, no = 0)
* Split the dataset into training and testing sets using `train_test_split()`

###  **Model Building:**

* Built a **Decision Tree Classifier** using `DecisionTreeClassifier()` from Scikit-learn
* Trained the model on 80% of the data and tested on the remaining 20%
* Fine-tuned hyperparameters like `max_depth` and `min_samples_split` to prevent overfitting

###  **Model Evaluation:**

* Calculated **Accuracy**, **Precision**, **Recall**, and **F1-score**
* Printed classification report and confusion matrix to assess model performance

###  **Tree Visualization:**

* Plotted the entire decision tree using `plot_tree()` to interpret the decision-making flow
* Analyzed **feature importances** to understand which factors influenced predictions the most

---

## 📈 **Visualizations:**

###  Target Distribution:

* Bar plot to display class imbalance in the target variable (`y`)
* Helps in identifying the need for stratified splitting or re-sampling techniques

###  Feature Importance Chart:

* Horizontal bar chart showcasing the top 10 features influencing the outcome
* Based on `feature_importances_` from the trained model

###  Decision Tree Diagram:

* Complete visualization of the trained decision tree using `plot_tree()`
* Shows nodes, conditions, and predicted outputs

---

##  **Learning Outcomes:**

* Gained hands-on experience in building and tuning a Decision Tree model
* Learned how to preprocess categorical features for machine learning
* Developed an understanding of how marketing and customer behavior data influence sales outcomes
* Visualized decision rules and feature contributions using interpretable ML tools
* Practiced structuring supervised ML workflows from data loading to model evaluation

---

##  **Author:**

**Yuva Sri**
*Data Science Student, VIT Vellore*
**Year:** 2025

---

Would you like me to generate this as a `.md` file or copy it into a notebook cell for direct use? I can also include GitHub badges, collapsible sections, or a table of contents if you'd like a more advanced version.
