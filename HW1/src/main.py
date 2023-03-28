import time
import nltk

from preprossing import Prepossing
from util import corss_validation, test_prediction

nltk.download('words')
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')

TRAIN_DATA_LOC = './data/1629732045_4949543_train_new.txt'
TEST_DATA_LOC = './data/1629732045_8755195_test_new.txt'
start_time = time.time()

def main():
    exit_loop = False
    while not exit_loop:
        print("Please select a options")
        print("1. Prepossessed data set")
        print("2. Perform cross validation on training data set")
        print("3. Perform data prediction on test set")
        print("4. Above all in sequence")
        print("5. Exit")
        choice = input("Enter your choice: ")

        if not choice.isnumeric():
            print("Oops, Please try again")
            continue

        choice = int(choice)

        if choice == 1:
            Prepossing(TRAIN_DATA_LOC, 'processed_train_data', 2)
            Prepossing(TEST_DATA_LOC, 'processed_test_data', 1)

        elif choice == 2:
            corss_validation()
        elif choice == 3:
            test_prediction()
        elif choice == 4:
            Prepossing(TRAIN_DATA_LOC, 'processed_train_data', 2)
            Prepossing(TEST_DATA_LOC, 'processed_test_data', 1)
            corss_validation()
            test_prediction()
        elif choice == 5:
            exit_loop = True
        else:
            print("Oops, Please try again")
    return
main()
