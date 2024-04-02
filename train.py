import torch
import time
import argparse
import os
from data_loader import load_data 
from parameters import DATASET, TRAINING, HYPERPARAMS, NETWORK
from model import build_model

import torch.nn as nn
import torch.optim as optim


def train(optimizer=HYPERPARAMS.optimizer, optimizer_param=HYPERPARAMS.optimizer_param, 
        learning_rate=HYPERPARAMS.learning_rate, keep_prob=HYPERPARAMS.keep_prob, 
        learning_rate_decay=HYPERPARAMS.learning_rate_decay, decay_step=HYPERPARAMS.decay_step,
        train_model=True):

        print("loading dataset " + DATASET.name + "...")
        if train_model:
                data, validation = load_data(validation=True)
        else:
                data, validation, test = load_data(validation=True, test=True)

        print("building model...")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        network = build_model()
        model = network.to(device)

        if train_model:
                # Training phase
                print("start training...")
                print("  - emotions = {}".format(NETWORK.output_size))
                print("  - model = {}".format(NETWORK.model))
                print("  - optimizer = '{}'".format(optimizer))
                print("  - learning_rate = {}".format(learning_rate))
                print("  - learning_rate_decay = {}".format(learning_rate_decay))
                print("  - otimizer_param ({}) = {}".format('beta1' if optimizer == 'adam' else 'momentum', optimizer_param))
                print("  - keep_prob = {}".format(keep_prob))
                print("  - epochs = {}".format(TRAINING.epochs))
                print("  - use landmarks = {}".format(NETWORK.use_landmarks))
                print("  - use hog + landmarks = {}".format(NETWORK.use_hog_and_landmarks))
                print("  - use hog sliding window + landmarks = {}".format(NETWORK.use_hog_sliding_window_and_landmarks))
                print("  - use batchnorm after conv = {}".format(NETWORK.use_batchnorm_after_conv_layers))
                print("  - use batchnorm after fc = {}".format(NETWORK.use_batchnorm_after_fully_connected_layers))

                criterion = nn.CrossEntropyLoss()
                optimizer = optim.Adam(model.parameters(), lr=learning_rate)

                start_time = time.time()
                for epoch in range(TRAINING.epochs):
                    model.train()
                    optimizer.zero_grad()

                    if NETWORK.use_landmarks:
                        inputs = [torch.tensor(data['X']).to(device), torch.tensor(data['X2']).to(device)]
                    else:
                        inputs = torch.tensor(data['X']).to(device)
                    labels = torch.tensor(data['Y']).to(device)

                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()

                    if epoch % TRAINING.snapshot_step == 0:
                        model.eval()
                        with torch.no_grad():
                            if NETWORK.use_landmarks:
                                validation_inputs = [torch.tensor(validation['X']).to(device), torch.tensor(validation['X2']).to(device)]
                            else:
                                validation_inputs = torch.tensor(validation['X']).to(device)
                            validation_labels = torch.tensor(validation['Y']).to(device)

                            validation_outputs = model(validation_inputs)
                            validation_loss = criterion(validation_outputs, validation_labels)
                            accuracy = torch.sum(torch.argmax(validation_outputs, dim=1) == validation_labels).item() / len(validation_labels)

                            print("Epoch: {}, Loss: {:.4f}, Validation Loss: {:.4f}, Accuracy: {:.2f}%".format(epoch, loss.item(), validation_loss.item(), accuracy * 100))

                training_time = time.time() - start_time
                print("training time = {0:.1f} sec".format(training_time))

                if TRAINING.save_model:
                    print("saving model...")
                    torch.save(model.state_dict(), TRAINING.save_model_path)

                print("evaluating...")
                validation_accuracy = evaluate(model, validation, device)
                print("  - validation accuracy = {0:.1f}".format(validation_accuracy * 100))
                return validation_accuracy
        else:
                # Testing phase : load saved model and evaluate on test dataset
                print("start evaluation...")
                print("loading pretrained model...")
                if os.path.isfile(TRAINING.save_model_path):
                        model.load_state_dict(torch.load(TRAINING.save_model_path))
                else:
                        print("Error: file '{}' not found".format(TRAINING.save_model_path))
                        exit()
                
                if not NETWORK.use_landmarks:
                        validation['X2'] = None
                        test['X2'] = None

                print("--")
                print("Validation samples: {}".format(len(validation['Y'])))
                print("Test samples: {}".format(len(test['Y'])))
                print("--")
                print("evaluating...")
                start_time = time.time()
                validation_accuracy = evaluate(model, validation, device)
                print("  - validation accuracy = {0:.1f}".format(validation_accuracy * 100))
                test_accuracy = evaluate(model, test, device)
                print("  - test accuracy = {0:.1f}".format(test_accuracy * 100))
                print("  - evalution time = {0:.1f} sec".format(time.time() - start_time))
                return test_accuracy

def evaluate(model, data, device):
        model.eval()
        with torch.no_grad():
            if NETWORK.use_landmarks:
                inputs = [torch.tensor(data['X']).to(device), torch.tensor(data['X2']).to(device)]
            else:
                inputs = torch.tensor(data['X']).to(device)
            labels = torch.tensor(data['Y']).to(device)

            outputs = model(inputs)
            accuracy = torch.sum(torch.argmax(outputs, dim=1) == labels).item() / len(labels)

        return accuracy

# parse arg to see if we need to launch training now or not yet
parser = argparse.ArgumentParser()
parser.add_argument("-t", "--train", default="no", help="if 'yes', launch training from command line")
parser.add_argument("-e", "--evaluate", default="no", help="if 'yes', launch evaluation on test dataset")
parser.add_argument("-m", "--max_evals", help="Maximum number of evaluations during hyperparameters search")
args = parser.parse_args()
if args.train=="yes" or args.train=="Yes" or args.train=="YES":
        train()
if args.evaluate=="yes" or args.evaluate=="Yes" or args.evaluate=="YES":
        train(train_model=False)
