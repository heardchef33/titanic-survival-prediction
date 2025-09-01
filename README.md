# Kaggle Titanic Survival Challenge 🚢

This repository documents my iterative approach to solving the classic Kaggle competition: "[Titanic - Machine Learning from Disaster](https://www.kaggle.com/c/titanic)" as well as my journey as I learn and correct my machine learning techniques. In this competition, the goal is to predict whether a passenger survived the sinking of the Titanic based on a set of personal and travel data.

The project is structured as a series of attempts, with each Jupyter notebook building upon the learnings of the last. This demonstrates a progression from a simple baseline to more complex techniques involving advanced imputation and feature engineering.

---

## 📊 Methodology & Notebooks

My approach was to start simple and incrementally add complexity, carefully observing the impact of each change on the model's performance.

### Notebook 1: `1_baseline_predictions.ipynb`
* **Objective**: Establish a performance benchmark.
* **Techniques**:
    * **Imputation**: I used basic, straightforward methods to handle missing values (e.g., filling missing ages with the mean/median and missing embarkment ports with the mode).
    * **Models**: Implemented baseline models like **Logistic Regression** to get an initial score.
* **Outcome**: This notebook provides the foundational score that all future experiments are measured against.

### Notebook 2: `2_advanced_imputation.ipynb`
* **Objective**: Improve the model's performance by using a more sophisticated imputation strategy.
* **Techniques**:
    * **Imputation**: Replaced the simple imputer with `IterativeImputer` (Multivariate Imputation) to predict missing `Age` values based on the relationships with other features.
    * **Models**: Utilized a more powerful ensemble model, **Random Forest**, to capture more complex patterns in the data.
* **Outcome**: This attempt aimed to see if a better imputation technique and a stronger model could lift the baseline score.

### Notebook 3: `3_feature_engineering.ipynb`
* **Objective**: Achieve the highest possible score by creating new, meaningful features from the existing data.
* **Techniques**:
    * **Feature Engineering**: Created new features such as `FamilySize` (from `SibSp` and `Parch`), `IsAlone`, and extracted `Title` (e.g., 'Mr.', 'Mrs.', 'Miss') from the `Name` column.
    * **Imputation**: Used the newly created `Title` feature to perform a more logical and accurate **manual imputation** of missing `Age` values (e.g., the average age of all passengers with the title 'Master').
    * **Model Exploration**: Experimented with several different classification models to find the best performer for the newly engineered feature set.

---

## 💡 Key Learnings

* **Importance of Feature Engineering**: Performance improvements came from creating new features, demonstrating that a deep understanding of the data is often more valuable than model complexity alone.
* **Iterative Improvement**: Starting with a simple baseline is crucial for measuring the true impact of more advanced techniques.
* **Contextual Imputation**: Using domain-specific logic (like titles for predicting age) can be far more effective than purely statistical imputation methods.
* **The Importance of Intentionality in Experimenting**: There were many occasions where I could not replicate good scores using pipelines from experimentation because I was sloppy with experimentation and the good scores came from data leakage.
* **Non-Linearity of Improvements**: Sometimes improvements I thought that would lead to model improvements actually decreased the score. This highlights the importance of understanding what is going on under the hood and resilience

## Summary 

I have learnt alot from this project. Even though my scores are not the highest I am glad I tried and learnt many different things such as pipelines, importance of avoiding data leakage, feature engineering, ensemble methods and advanced imputation. 