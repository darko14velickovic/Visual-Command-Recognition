
from __future__ import division, print_function, absolute_import

import tflearn
from tflearn import local_response_normalization
from tflearn.data_utils import shuffle, to_categorical
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.estimator import regression
from tflearn.data_preprocessing import ImagePreprocessing
from tflearn.data_augmentation import ImageAugmentation
import cv2
import numpy as np
import tensorflow as tf
import glob, os
from tflearn.datasets import cifar10


class CnnTrainer:

    def createNetwork(self, input_size, output_size):

        # Real-time data preprocessing
        img_prep = ImagePreprocessing()
        img_prep.add_featurewise_zero_center()
        img_prep.add_featurewise_stdnorm()

        # Real-time data augmentation
        img_aug = ImageAugmentation()
        img_aug.add_random_flip_leftright()
        img_aug.add_random_rotation(max_angle=25.)

        # Convolutional network building

        network = input_data(shape=[None, input_size, input_size, 3],
        data_preprocessing=img_prep,
        data_augmentation=img_aug)

        # network = input_data(shape=[None, input_size, input_size, 3])
        network = conv_2d(network, input_size + 20, 5, activation='relu')
        network = local_response_normalization(network)

        network = max_pool_2d(network, 2)

        network = conv_2d(network, input_size * 2, 5, activation='relu')
        network = local_response_normalization(network)

        network = max_pool_2d(network, 2)


        # added
        #network = conv_2d(network, input_size + 25, 5, activation='relu')

        # added
        #network = conv_2d(network, input_size + 25, 3, activation='relu')

        # ------------------------------------------
        network = conv_2d(network, input_size, 3, activation='relu')
        network = local_response_normalization(network)

        network = fully_connected(network, 256, activation='relu')
        network = dropout(network, 0.5)

        network = fully_connected(network, output_size, activation='softmax')
        network = regression(network, optimizer='adam',
                             loss='categorical_crossentropy',
                             learning_rate=0.001)
        return network

    def __init__(self, input_size, output_size, name, loadOnInit = False):

        self.version = '1.0'

        self.network = self.createNetwork(input_size, output_size)

        # Train using classifier
        self.model = tflearn.DNN(self.network, tensorboard_verbose=3, tensorboard_dir='model/tflearn_logs/')

        # If flag set try to load, if it doesn't load shit hits the fan... look into this.
        if(loadOnInit):
            try:
                self.model.load("model/" + name + '.tfl')

                self.modelLoaded = True
                self.modelTrained = True
            except:
                print("Failed to load model with name: " + name + ". Learning needed.")

                self.modelTrained = False
                self.modelLoaded = False
        else:
            self.modelTrained = False
            self.modelLoaded = False

    def evaluate_img(self, image):
        if self.modelTrained is False:
            print ("Model not trained!")
            return [0, 1, 0]
        return self.model.predict([image])

    def rename(self, dir, pattern):
        counter = 1

        # find max
        for pathAndFilename in glob.iglob(os.path.join(dir, pattern)):
            title, ext = os.path.splitext(os.path.basename(pathAndFilename))
            os.rename(pathAndFilename,
                      os.path.join(dir, "Kappa" + str(counter) + ext))
            counter += 1

        counter = 1
        for pathAndFilename in glob.iglob(os.path.join(dir, pattern)):
            os.rename(pathAndFilename,
                      os.path.join(dir, str(counter) + ext))
            counter += 1

    def loadImages(self, folders, start_index, example_count, test_count):
        #
        """
        Loads from folders test data and training data and returns images
        :return: First tuple of Training set and Training result set & tuple of Test set and Test results
        """

        # self.renaming(True)
        # self.renaming(False)

        # training = tuple()
        training_images = []
        training_labels = []

        # testing = tuple()
        testing_images = []
        testing_labels = []

        # add loading from the folders
        #folders = ['s1', 's2', 's3']
        label_counter = 0

        for folder in folders:
            for i in range(start_index, example_count):
                image = cv2.imread('training data/' + folder + '/' + str(i) + '.png', 1)
                new = np.divide(image, 1000.)
                training_images.append(new)
                training_labels.append(label_counter)

            for i in range(1, test_count + 1):
                image = cv2.imread('test data/' + folder + '/' + str(i) + '.png', 1)
                new = np.divide(image, 1000.)
                testing_images.append(new)
                testing_labels.append(label_counter)

            label_counter += 1

        # training.__add__(training_images)
        # training.__add__(training_labels)

        # testing.__add__(testing_images)
        # testing.__add__(testing_labels)

        return (np.array(training_images), np.array(training_labels)), (
        np.array(testing_images), np.array(testing_labels))

    # parse data from folder test


    def renaming(self, trainingFlag):
        if trainingFlag:
            self.rename(r'training data/s1/', r'*.png')
            self.rename(r'training data/s2/', r'*.png')
            self.rename(r'training data/s3/', r'*.png')
        else:
            self.rename(r'test data/s1/', r'*.png')
            self.rename(r'test data/s2/', r'*.png')
            self.rename(r'test data/s3/', r'*.png')

    def tf_learn(self, modelName, folders, start_index, end_index, test_count):
        #if(self.modelTrained):
        #print("Model already trained!")
        #return

        print("Starting batch training...")
        (X, Y), (X_test, Y_test) = self.loadImages(folders, start_index, end_index + 1, test_count)

        X, Y = shuffle(X, Y)
        Y = to_categorical(Y, len(folders))
        Y_test = to_categorical(Y_test, len(folders))

        self.model.fit(X, Y, n_epoch=50, shuffle=True, validation_set=(X_test, Y_test),
                       show_metric=True, batch_size=96, run_id=modelName)

        self.model.save("model/" + modelName + ".tfl")

        self.modelTrained = True
        return None



    def testing(self, folders):


        predictionLabel = 0

        confusion_matrix = []
        print("-------- Confusion matrix ---------")
        for folder in folders:
            correct_predictions = 0
            wrong_predictions = 0
            totalTests = 0

            confusion_row = np.zeros(folders.__len__())

            for pathAndFilename in glob.iglob(os.path.join("test data/" + folder + "/", r"*.png")):

                image = cv2.imread(pathAndFilename)
                totalTests += 1
                image = np.divide(image, 1000.)
                prediction = self.evaluate_img(image)
                maxIndex = np.argmax(prediction)

                confusion_row[maxIndex] += 1

                if maxIndex == predictionLabel:
                    correct_predictions += 1

                else:
                    wrong_predictions += 1

            print_string = folders[predictionLabel] + " | "

            confusion_matrix.append(confusion_row)

            for each in confusion_row:
                print_string += str(each) + " | "

            print(print_string)
            predictionLabel += 1

        print("--------- Metrics ---------")
        confusion_matrix = np.array(confusion_matrix)
        sum_total_gold = np.sum(confusion_matrix, axis=0)
        sum_total_predicted = np.sum(confusion_matrix, axis=1)

        print(sum_total_gold)
        print(sum_total_predicted)

        for i in range(0, folders.__len__()):
            tp = confusion_matrix[i][i]
            total_predicted = sum_total_predicted[i]
            total_gold = sum_total_gold[i]

            precision = tp / total_predicted
            recall = tp / total_gold
            fm = 2 * (precision * recall) / (precision + recall)

            print("Precision for class: " + folders[i]+ " is: " + str(precision))
            print("Recall for class: " + folders[i] + " is: " + str(recall))
            print("F-measure for class: " + folders[i] + " is: " + str(fm))
            print("---------------------------")

    def load_cnn(self, file_name):
        self.model.load("model/" +file_name+".tfl")
        self.modelTrained = True
