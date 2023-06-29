import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.ensemble import GradientBoostingClassifier
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedKFold, GridSearchCV


def load_data(
        train_data_path: str,
        test_data_path: str
):
    """
    Load data from CSV to pandas dataframe.
    :param train_data_path: Path of the training dataset
    :param test_data_path: Path of the testing dataset
    :return: Train and test dataframes.
    """
    return pd.read_csv(train_data_path), pd.read_csv(test_data_path)


def preprocessing_data(
        train_data: pd.DataFrame,
        test_data: pd.DataFrame
):
    # Drop useless columns
    train_data.drop(["Name", "Ticket", "Cabin"], axis=1, inplace=True)
    test_data.drop(["Name", "Ticket", "Cabin"], axis=1, inplace=True)

    # Encode some variables for the heatmap
    for i in ["Sex", "Embarked"]:
        le = preprocessing.LabelEncoder()
        train_data[i] = le.fit_transform(train_data[i])
        test_data[i] = le.transform(test_data[i])


def heat_map(train_data: pd.DataFrame):
    # Create a correlation matrix
    corr_matrix = train_data.corr()

    # Create a heatmap using seaborn
    plt.figure(figsize=(8, 6))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')

    # Set the title and display the plot
    plt.title('Correlation Matrix Heatmap')
    plt.savefig("heatmap.svg", format="svg")


def grid_search_cv(
    train_data: pd.DataFrame,
    test_data: pd.DataFrame
):
    x, x_test, y = get_x_y(train_data, test_data)

    model = GradientBoostingClassifier()
    kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=7)
    param_grid = {
        "learning_rate": [0.001, 0.005, 0.01, 0.05, 0.1, 0.2, 0.3],
        "n_estimators": [50, 100, 200, 300, 400, 500, 800]
    }
    # Perform the gird search
    grid_search = GridSearchCV(model, param_grid, scoring="neg_log_loss", n_jobs=-1, cv=kfold)
    grid_result = grid_search.fit(x, y)

    means = grid_result.cv_results_['mean_test_score']
    scores = np.array(means).reshape(len(param_grid["learning_rate"]), len(param_grid["n_estimators"]))

    # Plot the neg_log_loss of each combination
    plt.clf()
    for i, value in enumerate(param_grid["learning_rate"]):
        plt.plot(param_grid["n_estimators"], scores[i], label='learning_rate: ' + str(value))
    plt.legend()
    plt.xlabel('n_estimators')
    plt.ylabel('Log Loss')
    plt.savefig('hyperparameters_comparison.svg', format='svg')


def get_x_y(
    train_data: pd.DataFrame,
    test_data: pd.DataFrame
):
    # Finalize the data
    x = pd.get_dummies(train_data[["Pclass", "Sex", "Fare", "Embarked"]]).fillna(-1)
    x_test = pd.get_dummies(test_data[["Pclass", "Sex", "Fare", "Embarked"]]).fillna(-1)
    y = train_data["Survived"]
    return x, x_test, y


def build_train_model(
        train_data: pd.DataFrame,
        test_data: pd.DataFrame

):
    x, x_test, y = get_x_y(train_data, test_data)

    # Train the model
    model = GradientBoostingClassifier(
        learning_rate=0.01,
        n_estimators=500
    )
    model.fit(x, y)

    # Predict and store predictions
    predictions = model.predict(x_test)
    output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': predictions})
    output.to_csv('submission.csv', index=False)


def main():
    train_data, test_data = load_data("./data/train.csv", "./data/test.csv")
    preprocessing_data(train_data, test_data)
    heat_map(train_data)
    build_train_model(train_data, test_data)


if __name__ == "__main__":
    main()
