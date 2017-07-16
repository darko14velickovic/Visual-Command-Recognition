
from image_processor.Trainer import CnnTrainer
import glob, os
import sys, getopt
import cv2 as cv
import numpy as np


def str2bool(v):
  return v.lower() in ("yes", "true", "t", "1")

def main(argv):
    networkName = 'default'
    usage_string = 'ctrainer.py -d <image dimension DIM x DIM> -n <network name> -f <comma separated names of folders' \
                   'for training and testing> -ec <number of training images> -tc number of testing images'

    dimension = 40
    folders = []
    start_index = 1
    end_index = 50
    test_count = 5
    resume_flag = False
    learning = True
    try:
        opts, args = getopt.getopt(argv,"hd:n:f:s:e:t:r:l:")
    except getopt.GetoptError:
        print (usage_string)
        sys.exit(2)

    for opt, arg in opts:
        if opt == '-h':
            print (usage_string)
            sys.exit(0)
        elif opt in ("-d", "--dimension"):
            # print arg
            dimension = int(arg)
        elif opt in ("-n", "--name"):
            networkName = arg
        elif opt in ("-f", "--folders"):
            folders = arg.split(',')
        elif opt in ("-s", "--start-index"):
            start_index = int(arg)
        elif opt in ("-e", "--end-index"):
            end_index = int(arg)
        elif opt in ("-t", "--tests"):
            test_count = int(arg)
        elif opt in ("-r", "--resume"):
            resume_flag = str2bool(arg)
        elif opt in ("-l", "--learning"):
            learning = str2bool(arg)


    print ('Network name is ' + networkName)
    print ('Dimension of images is: ' + str(dimension) + ' x ' +str(dimension))
    print ('Folders for training and testing are: ')
    print (folders)
    print ('Start index for training is: ' + str(start_index))
    print ('End index for training is: ' + str(end_index))
    print ('Test count is: ' + str(test_count))
    print ("Resume tag is: " + str(resume_flag))

    if learning:

        trainer = CnnTrainer(dimension, folders.__len__(), networkName, resume_flag)
        trainer.tf_learn(networkName, folders, start_index, end_index, test_count)

    else:

        trainer = CnnTrainer(dimension, folders.__len__(), networkName, True)
        trainer.testing(folders)


if __name__ == "__main__":
   main(sys.argv[1:])


def mergeInOneImage(dir, pattern):
    imageNumber = 0
    imageRow = 0
    bigImage = np.zeros()
    for pathAndFilename in glob.iglob(os.path.join(dir, pattern)):
        # title, ext = os.path.splitext(os.path.basename(pathAndFilename))
        image = cv.imread(pathAndFilename)
        if imageNumber == 19:
            imageRow += 1
            imageNumber = 0



        imageNumber += 1
