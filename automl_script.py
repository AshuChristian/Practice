from tpot import TPOTClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

iris = load_iris()
x = iris.data
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(x,y, test_size = 0.2, random_state=42)

tpot = TPOTClassifier(
    generations = 5,
    population_size = 20,
    verbosity = 2,
    random_state = 42,
    config_dict = "TPOT sparse",
    memory = 'auto',
    n_jobs = -1,
    cv = 5
)

tpot.fit(X_train, y_train)

accuracy = tpot.score(X_test, y_test)
print(f"Accuracy: {accuracy:.2f}")

tpot.export('best_model_pipeleine.py')