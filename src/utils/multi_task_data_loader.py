import json
from typing import List, Tuple
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.preprocessing import LabelEncoder

class DataLoader:
    """
    The DataLoader class is a simple utility class that is used to load the article bodies
    and the corresponding keywords and sentiment from the given file.
    """

    def __init__(self, train_file_path, test_file_path):
        """
        Initializes the DataLoader with the given file path and instantiates the 
        MultiLabelBinarizer that is used to encode the keywords, along with the LabelEncoder
        that is used to encode the sentiment.

        Args:
            train_file_path (str): The path to the training file that contains the articles and keywords.
            test_file_path (str): The path to the test file that contains the articles and keywords.
        """
        self.train_file_path = train_file_path
        self.test_file_path = test_file_path
        self.multi_label_binarizer = MultiLabelBinarizer()
        self.label_encoder = LabelEncoder()
        self.class_labels = []

    def load_data(self) -> Tuple[List[List[str]], List[List[str]], List[str], List[List[str]], List[List[str]], List[str]]:
        """
        Loads the data from the file and returns the article bodies and the corresponding keywords and sentiment.
        The keywords are encoded for multi-label classification using the MultiLabelBinarizer while the sentiment
        is encoded for multi-class classification using the LabelEncoder.

        Returns:
            list: The list of train article bodies.
            list: The list of train corresponding keywords, encoded for multi-label classification.
            list: The list of train sentiment labels, encoded for multi-class classification.
            list: The list of test article bodies.
            list: The list of test corresponding keywords, encoded for multi-label classification.
            list: The list of test sentiment labels, encoded for multi-class classification.

        Raises:
            ValueError: If the file path is not set.
        """
        if not self.train_file_path:
            raise ValueError('DataLoader: train_file_path is not set')
        if not self.test_file_path:
            raise ValueError('DataLoader: test_file_path is not set')

        # Load both datasets and combine their keywords to fit the MultiLabelBinarizer
        train_data, train_labels, train_sentiment = self._load_data(self.train_file_path)
        test_data, test_labels, test_sentiment = self._load_data(self.test_file_path)

        # The train_labels and test_labels are lists of lists, where each inner list contains the keywords
        # for the corresponding article. We need to combine these lists to fit the MultiLabelBinarizer.
        all_labels = train_labels + test_labels

        # Combine the sentiment labels of the train and test data
        all_sentiment = train_sentiment + test_sentiment

        # Fit the MultiLabelBinarizer to the combined list of keywords
        self.multi_label_binarizer.fit(all_labels)
        self.label_encoder.fit(all_sentiment)

        # Transform the train and test labels to the multi-label format
        train_labels = self.multi_label_binarizer.transform(train_labels)
        test_labels = self.multi_label_binarizer.transform(test_labels)

        # Transform the train and test sentiment labels
        train_sentiment = self.label_encoder.transform(train_sentiment)
        test_sentiment = self.label_encoder.transform(test_sentiment)

        return train_data, train_labels, train_sentiment, test_data, test_labels, test_sentiment

    def get_label_classes(self) -> List[str]:
        """
        Returns the list of label classes.

        Returns:
            list: The list of label classes.
        """
        return self.multi_label_binarizer.classes_
    
    def get_classes_sentiment(self) -> List[str]:
        """
        Returns the list of sentiment classes.

        Returns:
            list: The list of sentiment classes.
        """
        return self.label_encoder.classes_

    def _load_data(self, file_path: str) -> Tuple[List[List[str]], List[List[str]], List[str]]:
        """
        Loads the data from the file and returns the article bodies and the corresponding keywords.
        The article bodies are chunked into smaller parts that can be handled by the SentenceTransformer.

        Args:
            file_path (str): The path to the file that contains the articles and keywords.

        Returns:
            list: The list of article bodies.
            list: The list of corresponding keywords.
        """
        with open(file_path, 'r', encoding='utf-8') as file:
            data = [json.loads(line) for line in file.readlines()]
            
            articles = [self._chunk_data(article['body']) for article in data]
            keywords = [article['keywords'].lower().split(';') for article in data]
            sentiment = [article['sentiment'].lower() for article in data]

        return articles, keywords, sentiment
    
    def _chunk_data(self, data: str) -> str:
        """
        Chunks the given data into smaller parts that can be handled by the SentenceTransformer.

        Args:
            data (str): The data that should be chunked.

        Returns:
            str: The chunked data.
        """
        words = data.split(' ')
        chunks = []
        chunk = []
        chunk_length = 0
        for word in words:
            word_length = len(word)
            if chunk_length + word_length < 512:
                chunk.append(word)
                chunk_length += word_length + 1  # +1 for the space
            else:
                chunks.append(' '.join(chunk))
                chunk = [word]
                chunk_length = word_length
        if chunk:
            chunks.append(' '.join(chunk))

        return chunks
