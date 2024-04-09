import json
from typing import List, Tuple

class DataLoader:
    """
    The DataLoader class is a simple utility class that is used to load the article bodies
    and the corresponding keywords from the given file.
    """

    def __init__(self, file_path):
        """
        Initializes the DataLoader with the given file path.

        Args:
            file_path (str): The path to the file that contains the articles and keywords.
        """

        self.file_path = file_path

    def load_data(self) -> Tuple[List[str], List[List[str]]]:
        """
        Loads the data from the file and returns the article bodies and the corresponding keywords.

        Returns:
            list: The list of article bodies.
            list: The list of corresponding keywords.
        """
        
        f = open(self.file_path, 'r', encoding='utf-8')

        data = []
        labels = []

        for line in f.readlines():
            article = json.loads(line)
            data.append(article['body'])
            labels.append(article['keywords'])

        f.close()

        return data, labels
    