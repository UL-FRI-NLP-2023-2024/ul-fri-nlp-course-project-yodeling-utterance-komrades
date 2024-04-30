import numpy as np
import torch
from sklearn.metrics import f1_score, recall_score, precision_score

from networks.baseline_classifier import BaselineClassifier
from utils.baseline_utils import get_criterion, get_optimizer, get_scheduler
from utils.data_loader import DataLoader

config = {
    #### Classifier parameters
    # - model_name: The name of the Sentence Transformer model that should be used to generate embeddings.
    # - num_hidden_layers: The number of hidden layers in the classifier.
    # - hidden_layer_size: The size of the hidden layers in the classifier.
    # - dropout_rate: The dropout rate of the classifier.
    # - freeze_automodel: Whether the Sentence Transformer model should be frozen during training.
    # - activation_function: The activation function of the hidden layers. One of 'relu', 'tanh', 'sigmoid'.
    'model_name': 'distiluse-base-multilingual-cased',
    'num_hidden_layers': 2,
    'hidden_layer_size': 128,
    'dropout_rate': 0.1,
    'freeze_automodel': False,
    'activation_function': 'relu',

    #### Training parameters
    # - loss_function: The loss function used for training. One of 'bce_with_logits', 'cross_entropy', 'multi_label_soft_margin', 'multi_label_margin'.
    # - optimizer: The optimizer used for training. One of 'adam', 'sgd'.
    # - learning_rate: The learning rate of the optimizer.
    # - scheduler: The learning rate scheduler. One of 'none', 'linear', 'exponential'.
    # - scheduler_gamma: The gamma parameter of the learning rate scheduler.
    # - scheduler_step: The step parameter of the learning rate scheduler. Only used for the 'linear' scheduler.
    # - epochs: The number of epochs to train the model.
    'loss_function': 'bce_with_logits',
    'optimizer': 'adam',
    'learning_rate': 0.001,
    'scheduler': 'linear',
    'scheduler_gamma': 0.1,
    'scheduler_step': 1,
    'epochs': 5,

    #### Testing parameters
    # - threshold: The threshold used for the predictions. If the probability of a class is higher than the threshold, the class is predicted.
    'threshold': 0.533,

    #### General parameters
    # - verbose: Whether to print additional information during training/testing.
    # - device: The device to use for training/testing. One of 'cuda', 'cpu'.
    # - mode: The mode of the script. One of 'train', 'test'.
    # - save_model: Whether to save the model after training.
    # - saved_model_name: The name of the saved model file.
    'verbose': True,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'mode': 'train',
    'save_model': True,
    'saved_model_name': 'baseline_classifier',
}

def train():
    """
    Performs the training of the BaselineClassifier model. The training is performed on the Slovenian
    SentiNews dataset. The model is first trained, and then evaluated on the test set. The model is saved
    after training if the 'save_model' parameter is set to True in the configuration.

    The entire training/evaluation/saving process is controlled by the configuration dictionary.

    Raises:
        ValueError: If the model name is not specified in the configuration.
    """
    data_loader = DataLoader('../data/SentiNews/slovenian_train.json', '../data/SentiNews/slovenian_test.json')
    train_data, train_labels, test_data, test_labels = data_loader.load_data()

    if config['verbose']:
        print('Train: Loaded train dataset.')

    config['output_classes'] = len(data_loader.get_label_classes())

    classifier = BaselineClassifier(config['model_name'], config)
    classifier.set_mode('train')

    if config['verbose']:
        print('Train: Initialized classifier.')

    criterion = get_criterion(config)
    optimizer = get_optimizer(config, classifier)
    scheduler = get_scheduler(config, optimizer)

    if config['verbose']:
        print('Train: Initialized criterion, optimizer and scheduler.')
        print('Train: Starting training...')

    for epoch in range(config['epochs']):
        for i in range(len(train_data)):
            inputs = train_data[i]
            label = train_labels[i]

            label = torch.tensor(label).to(config['device'])

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = classifier(inputs)
            label = label.view(outputs.shape).float()
            loss = criterion(outputs, label)

            # Backward pass
            loss.backward()
            optimizer.step()

            if scheduler:
                scheduler.step()
        print('Epoch: {}, Loss: {}'.format(epoch + 1, loss.item()))

        if epoch % 5 == 0:
            classifier.set_mode('eval')
            _test(classifier, test_data, test_labels, data_loader.get_label_classes())
            classifier.set_mode('train')

    if config['verbose']:
        print('Train: Finished training.')

    # Evaluate the model
    classifier.set_mode('eval')

    _test(classifier, test_data, test_labels, data_loader.get_label_classes())

    if config['verbose']:
        print('Train: Finished testing.')

    # Save the model
    if config['save_model']:
        if config['model_name']:
            if config['verbose']:
                print('Train: Saving model.')
            torch.save(classifier.state_dict(), '../models/{}.pt'.format(config['saved_model_name']))
        else:
            raise ValueError('Cannot save the model: Model name not specified')


def _test(classifier: BaselineClassifier, test_data: list, test_labels: list, label_classes: list):
    """
    Tests the provided classifier on the test data and prints the accuracy of the model. This 
    function is used to evaluate the model during training and during testing.

    Args:
        classifier (BaselineClassifier): The classifier model.
        test_data (list): The test data.
        test_labels (list): The test labels.
    """
    with torch.no_grad():
        predictions = []
        true_labels = []
        for i in range(len(test_data)):
            inputs = test_data[i]
            label = test_labels[i]

            # Convert the inputs and labels to tensors
            #inputs = torch.tensor(inputs).to(config['device'])
            label = torch.tensor(label).to(config['device'])

            # Forward pass
            outputs = classifier(inputs)

            # Convert the outputs to probabilities and predictions.
            probs = torch.sigmoid(outputs)
            preds = (probs > config['threshold']).float()

            # if config['verbose']:
            #     # Print the 10 highest probabilities in the probs tensor
            #     print('Highest probabilities: {}'.format(probs.squeeze().topk(10).values.cpu().numpy()))

            predictions.append(preds.squeeze().cpu().numpy())
            true_labels.append(label.cpu().numpy())

            # if config['verbose']:
            #     predicted_labels = [label_classes[i] for i in range(len(predictions[-1])) if predictions[-1][i].item() == 1]
            #     true_labels = [label_classes[i] for i in range(len(true_labels[-1])) if true_labels[-1][i].item() == 1]
# 
            #     print('Predicted labels: {}'.format(predicted_labels))
            #     print('True labels: {}'.format(true_labels))

        predictions = np.array(predictions)
        true_labels = np.array(true_labels)

        print('Model evaluation:')
        print('     F1 Score: {}'.format(f1_score(true_labels, predictions, average='samples', zero_division=0)))
        print('     Recall: {}'.format(recall_score(true_labels, predictions, average='samples', zero_division=0)))
        print('     Precision: {}'.format(precision_score(true_labels, predictions, average='samples', zero_division=0)))

def test():
    """
    Tests the BaselineClassifier model on the Slovenian SentiNews test dataset. The model is loaded from the
    saved model file and is evaluated on the test set. The accuracy of the model is printed to the console.
    """
    data_loader = DataLoader('../data/SentiNews/slovenian_train.json', '../data/SentiNews/slovenian_test.json')
    train_data, train_labels, test_data, test_labels = data_loader.load_data()

    if config['verbose']:
        print('Test: Loaded test dataset.')

    config['output_classes'] = len(data_loader.get_label_classes())

    classifier = BaselineClassifier(config['model_name'], config)
    classifier.load_state_dict(torch.load('../models/{}.pt'.format(config['model_name'])))
    classifier.set_mode('eval')

    if config['verbose']:
        print('Test: Loaded model.')
        print('Test: Starting testing...')

    _test(classifier, test_data, test_labels, data_loader.get_label_classes())

    if config['verbose']:
        print('Test: Finished testing.')


if __name__ == '__main__':
    if config['mode'] == 'train':
        train()
    elif config['mode'] == 'test':
        test()
    else:
        raise ValueError('Invalid mode: {}'.format(config['mode']))

