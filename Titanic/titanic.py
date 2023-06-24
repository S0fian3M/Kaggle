import pandas as pd
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import seaborn as sns


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


def heat_map(
        train_data: pd.DataFrame,
        save: bool = True
):
    # Create a correlation matrix
    corr_matrix = train_data.corr()

    # Create a heatmap using seaborn
    plt.figure(figsize=(8, 6))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')

    # Set the title and display the plot
    plt.title('Correlation Matrix Heatmap')
    if save:
        plt.savefig("heatmap.svg", format="svg")
    plt.show()


def train_model_and_predict(
        train_data: pd.DataFrame,
        test_data: pd.DataFrame
):
    # Finalize the data
    x = pd.get_dummies(train_data[["Pclass", "Sex", "Fare", "Embarked"]]).fillna(-1)
    x_test = pd.get_dummies(test_data[["Pclass", "Sex", "Fare", "Embarked"]]).fillna(-1)
    y = train_data["Survived"]

    # Train the model
    model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=1)
    model.fit(x, y)

    # Predict and store predictions
    predictions = model.predict(x_test)
    output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': predictions})
    output.to_csv('submission.csv', index=False)


def main():
    train_data, test_data = load_data("./data/train.csv", "./data/test.csv")
    preprocessing_data(train_data, test_data)
    heat_map(train_data)
    train_model_and_predict(train_data, test_data)


if __name__ == "__main__":
    main()
