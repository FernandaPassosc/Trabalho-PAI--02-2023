import pandas as pd
from sklearn.model_selection import train_test_split

class Classifications:
    def _init_(self, csv_file):
        self.data = pd.read_csv(csv_file)
        self.train_data, self.test_data = self._split_data()

    def _split_data(self, test_size=0.2, random_state=42):
        train_data = pd.DataFrame()
        test_data = pd.DataFrame()

        for class_label in self.data['bethesda_system'].unique():
            class_data = self.data[self.data['bethesda_system'] == class_label]
            class_train, class_test = train_test_split(class_data, test_size=test_size, random_state=random_state)
            train_data = pd.concat([train_data, class_train], axis=0)
            test_data = pd.concat([test_data, class_test], axis=0)

        return train_data, test_data

    def get_train_data(self):
        return self.train_data

    def get_test_data(self):
        return self.test_data

# Exemplo de uso
csv_file_path = 'classifications.csv'
classification_instance = Classifications(csv_file_path)

# Acesse os conjuntos de treino e teste
train_data = classification_instance.get_train_data()
test_data = classification_instance.get_test_data()