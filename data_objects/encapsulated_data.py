# initializing the respective data objects.
from sklearn.model_selection import train_test_split

class EmailDataSet:
    def __init__(self, X, y2, y3, y4, test_size=0.2, random_state=42):
        self.X_train, self.X_test = train_test_split(X, test_size=test_size, random_state=random_state)
        self.y2_train, self.y2_test = train_test_split(y2, test_size=test_size, random_state=random_state)
        self.y3_train, self.y3_test = train_test_split(y3, test_size=test_size, random_state=random_state)
        self.y4_train, self.y4_test = train_test_split(y4, test_size=test_size, random_state=random_state)
