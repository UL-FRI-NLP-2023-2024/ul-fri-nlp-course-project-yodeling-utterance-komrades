import torch
import torch.nn as nn

from transformers import AutoModel
from sentence_transformers import SentenceTransformer
from typing import Dict

class BaselineClassifier(nn.Module):
    """
    The BaselineClassifier is an implementation of a simple neural network used for classification. The
    BaselineClassifier is used as an addition to a Sentence Transformer model and is used to classify the 
    embeddings into the desired categories. As the embeddings can be classified into multiple categories,
    the BaselineClassifier is designed to be used for multi-label classification tasks.

    The BaselineClassifier consists of two main parts:
        1. A Sentence Transformer model that is used to generate embeddings of the input data.
        2. A simple feedforward neural network that is used to classify the embeddings into the desired categories.
    """

    def __init__(self, model_name: str, config: Dict[str, any]):
        """
        Initializes the BaselineClassifier with the given model name and configuration. The configuration
        contains the hyperparameters of the classifier, such as the number of hidden layers, the hidden layer
        size, the output classes, the dropout rate, etc.

        Args:
            model_name (str): The name of the Sentence Transformer model that should be used to generate embeddings.
            config (Dict[str, any]): The configuration of the BaselineClassifier.
        """
        super(BaselineClassifier, self).__init__()

        self._validate_config(config)
        self._extract_config(config)

        # Load the Sentence Transformer model and set it to training mode.
        self.auto_model = SentenceTransformer(model_name) # AutoModel.from_pretrained(model_name)
        self.auto_model.train()

        # Freeze the Sentence Transformer model if specified in the configuration. If the model is frozen,
        # the gradients of the model parameters will not be updated during training, which means that only
        # the classifier part of the BaselineClassifier will be trained.
        if self.freeze_automodel:
            for param in self.auto_model.parameters():
                param.requires_grad = False

        # Get the embedding dimensions of the Sentence Transformer model.
        self.embedding_dimensions = self.auto_model.get_sentence_embedding_dimension()

        # Assemble the classifier based on the configuration.
        layers = []
        for _ in range(self.num_hidden_layers):
            # The first layer should have the input dimensions equal to the embedding dimensions. All
            # other layers should have the input dimensions equal to the hidden layer size.
            if len(layers) == 0:
                layers.append(nn.Linear(self.embedding_dimensions, self.hidden_layer_size))
            else:
                layers.append(nn.Linear(self.hidden_layer_size, self.hidden_layer_size))

            # Add the activation function as specified in the configuration. If no activation function is
            # specified, the default activation function is ReLU.
            if 'activation_function' in config:
                if config['activation_function'] == 'relu':
                    layers.append(nn.ReLU())
                elif config['activation_function'] == 'tanh':
                    layers.append(nn.Tanh())
                elif config['activation_function'] == 'sigmoid':
                    layers.append(nn.Sigmoid())
            else:
                layers.append(nn.ReLU())
            
            # Add dropout if specified in the configuration.
            if self.dropout_rate:
                layers.append(nn.Dropout(self.dropout_rate))

        # Add the output layer that has the output dimensions equal to the number of output classes.
        layers.append(nn.Linear(self.hidden_layer_size, self.output_classes))
        self.classifier = nn.Sequential(*layers)    
        

    def forward(self, x: torch.Tensor) -> torch.Tensor:
    #def forward(self, input_ids=None, attention_mask=None, token_type_ids=None) -> torch.Tensor:
        """
        Performs a forward pass through the BaselineClassifier. The forward pass consists of passing the input
        through the Sentence Transformer model to generate embeddings, extracting the embeddings of the [CLS] token,
        passing the embeddings through the classifier to get the logits.

        Args:
            x (torch.Tensor): The input tensor that should be classified.

        Returns:
            torch.Tensor: The output tensor that contains the logits of the input tensor.
        """

        #outputs = self.auto_model(x)
        #x = self.auto_model.tokenize(x)
        #x = self.auto_model.encode(x)
        #x = torch.tensor(x).to(self.device)
        
        # inputs = self.auto_model.tokenizer(x, return_tensors="pt", padding=True, truncation=True)
        # inputs = {key: val.to(self.device) for key, val in inputs.items()}
        # outputs = self.auto_model(**inputs)

        #outputs = self.auto_model(x)

        # Extract the last hidden states from the output of the transformer.
        # These states contain rich contextual information about the input data.
        #last_hidden_state = outputs[0]

        # Extract the embeddings of the [CLS] token, which is a special token that
        # is added to the beginning of the input data, and is designed to capture
        # overall semantic information of the input sequence.
        # cls_embedding = last_hidden_state[:, 0, :]

        # Pass the embeddings of the [CLS] token through the classifier to get the logits.
        #logits = self.classifier(cls_embedding)

        inputs = [x]
        embeddings = self.auto_model.encode(inputs, convert_to_tensor=True)
        embeddings = embeddings.to(self.device)
        self.classifier = self.classifier.to(self.device)
        logits = self.classifier(embeddings)
        return logits
    
    def set_mode(self, mode: str):
        """
        Sets the mode of the BaselineClassifier. The mode can be either 'train' or 'eval'.
        This method should be used instead of calling self.train() and self.eval() directly, as it
        also sets the mode of the Sentence Transformer model.

        Args:
            mode (str): The mode that the BaselineClassifier should be set to. Supported modes are: train, eval.
        """
        if mode == 'train':
            self.train()
            self.auto_model.train()
        elif mode == 'eval':
            self.eval()
            self.auto_model.eval()
        else:
            raise ValueError('Invalid mode. Supported modes are: train, eval')


    def _validate_config(self, config: Dict[str, any]):
        """
        Validates the configuration of the BaselineClassifier. The configuration should contain the 'output_classes'
        field, which specifies the number of output classes that the classifier should classify the embeddings into.
        If the 'activation_function' field is present, it should be one of 'relu', 'tanh', 'sigmoid'.

        Args:
            config (Dict[str, any]): The configuration of the BaselineClassifier.

        Raises:
            ValueError: If the configuration is invalid.
        """
        if 'output_classes' not in config:
            raise ValueError('output_classes is a required field in the config')
        if 'activation_function' in config:
            if config['activation_function'] not in ['relu', 'tanh', 'sigmoid']:
                raise ValueError('Invalid activation function. Supported activation functions are: relu, tanh, sigmoid')
            
    def _extract_config(self, config: Dict[str, any]):
        """
        Extracts the configuration of the BaselineClassifier. The configuration should contain the 'output_classes',
        'dropout_rate', 'hidden_layer_size', 'num_hidden_layers', 'freeze_automodel', 'activation_function' fields.

        Args:
            config (Dict[str, any]): The configuration of the BaselineClassifier.
        """
        self.output_classes = config['output_classes']
        self.dropout_rate = config.get('dropout_rate')
        self.hidden_layer_size = config.get('hidden_layer_size', 512)
        self.num_hidden_layers = config.get('num_hidden_layers', 1)
        self.freeze_automodel = config.get('freeze_automodel', False)
        self.activation_function = config.get('activation_function')
        self.device = config.get('device', 'cpu')