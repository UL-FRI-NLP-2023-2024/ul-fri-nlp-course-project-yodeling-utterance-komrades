import torch
import torch.nn as nn

class Classifier(nn.Module):
    """
    The Classifier is an implementation of a simple neural network used for classification. The
    Classifier is used as an addition to the SBERT model and is used to classify the embeddings into 
    the desired categories. 
    """

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        """
        Initializes the Classifier with the given input, hidden and output dimensions. The Classifier
        is a simple feedforward neural network with a single hidden layer.

        Args:
            input_dim (int): The dimension of the input layer, this is the dimension of the SBERT embeddings.
            hidden_dim (int): The dimension of the hidden layer, this is the dimension of the hidden layer. 
                              Modifying this hyperparameter can allow the model to learn more complex patterns 
                              but can also lead to overfitting.
            output_dim (int): The dimension of the output layer, this is the number of classes that the model
                              should classify the embeddings into.
        """

        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.1)
        #self.softmax = nn.Softmax(dim=1)

    def forward(self, x) -> torch.Tensor:
        """
        Performs a forward pass through the Classifier. The forward pass consists of passing the input
        through the first linear layer, applying the ReLU activation function, applying dropout, passing
        the output through the second linear layer and applying the softmax activation function.

        Args:
            x (torch.Tensor): The input tensor that should be classified.

        Returns:
            torch.Tensor: The output tensor that contains the probabilities of the input tensor belonging
                          to each of the classes.
        """

        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        #x = self.softmax(x)
        return x