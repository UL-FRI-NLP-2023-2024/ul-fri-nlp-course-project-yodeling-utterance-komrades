import json
from typing import List, Tuple
from sklearn.preprocessing import MultiLabelBinarizer

class DataLoader:
    """
    The DataLoader class is a simple utility class that is used to load the article bodies
    and the corresponding keywords from the given file.
    """

    def __init__(self, train_file_path, test_file_path):
        """
        Initializes the DataLoader with the given file path and instantiates the 
        MultiLabelBinarizer that is used to encode the keywords.

        Args:
            train_file_path (str): The path to the training file that contains the articles and keywords.
            test_file_path (str): The path to the test file that contains the articles and keywords.
        """
        self.train_file_path = train_file_path
        self.test_file_path = test_file_path
        self.multi_label_binarizer = MultiLabelBinarizer()
        self.class_labels = []

    def load_data(self) -> Tuple[List[str], List[List[str]]]:
        """
        Loads the data from the file and returns the article bodies and the corresponding keywords.
        The keywords are encoded for multi-label classification using the MultiLabelBinarizer.

        Returns:
            list: The list of article bodies.
            list: The list of corresponding keywords, encoded for multi-label classification.

        Raises:
            ValueError: If the file path is not set.
        """
        if not self.train_file_path:
            raise ValueError('DataLoader: train_file_path is not set')
        if not self.test_file_path:
            raise ValueError('DataLoader: test_file_path is not set')

        # Load both datasets and combine their keywords to fit the MultiLabelBinarizer
        train_data, train_labels = self._load_data(self.train_file_path)
        test_data, test_labels = self._load_data(self.test_file_path)

        # The train_labels and test_labels are lists of lists, where each inner list contains the keywords
        # for the corresponding article. We need to combine these lists to fit the MultiLabelBinarizer.
        all_labels = train_labels + test_labels

        # Fit the MultiLabelBinarizer to the combined list of keywords
        self.multi_label_binarizer.fit(all_labels)

        # Transform the train and test labels to the multi-label format
        train_labels = self.multi_label_binarizer.transform(train_labels)
        test_labels = self.multi_label_binarizer.transform(test_labels)

        return train_data, train_labels, test_data, test_labels

    def get_label_classes(self) -> List[str]:
        """
        Returns the list of label classes.

        Returns:
            list: The list of label classes.
        """
        return self.multi_label_binarizer.classes_
    
    def _load_data(self, file_path: str) -> Tuple[List[str], List[List[str]]]:
        """
        Loads the data from the file and returns the article bodies and the corresponding keywords.

        Args:
            file_path (str): The path to the file that contains the articles and keywords.

        Returns:
            list: The list of article bodies.
            list: The list of corresponding keywords.
        """
        with open(file_path, 'r', encoding='utf-8') as file:
            data = [json.loads(line) for line in file.readlines()]
            
            articles = [article['body'] for article in data]
            keywords = [article['keywords'].lower().split(';') for article in data]

        return articles, keywords
