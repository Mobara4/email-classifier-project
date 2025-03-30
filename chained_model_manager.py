from models.random_forest_model import RandomForestModel
import numpy as np
from sklearn.preprocessing import LabelEncoder

class ChainedModelManager:
    def __init__(self, dataset):
        self.data = dataset
        self.model2 = RandomForestModel()
        self.model3 = RandomForestModel()
        self.model4 = RandomForestModel()

    def run(self):
        # Step 1 - Train on Type 2
        self.model2.train(self.data.X_train, self.data.y2_train)
        y2_pred = self.model2.predict(self.data.X_test)
        self.model2.print_results(self.data.y2_test, y2_pred)

        # Encode y2_pred before stacking
        le2 = LabelEncoder()
        y2_encoded = le2.fit_transform(y2_pred)

        # Step 2 - Train on Type 3 using encoded Type 2
        X_plus_y2 = np.column_stack((self.data.X_test, y2_encoded))
        self.model3.train(X_plus_y2, self.data.y3_test)
        y3_pred = self.model3.predict(X_plus_y2)
        self.model3.print_results(self.data.y3_test, y3_pred)

        # Encode y3_pred before stacking again
        le3 = LabelEncoder()
        y3_encoded = le3.fit_transform(y3_pred)

        # Step 3 - Train on Type 4 using encoded Type 2 + 3
        X_plus_y2y3 = np.column_stack((self.data.X_test, y2_encoded, y3_encoded))
        self.model4.train(X_plus_y2y3, self.data.y4_test)
        y4_pred = self.model4.predict(X_plus_y2y3)
        self.model4.print_results(self.data.y4_test, y4_pred)
