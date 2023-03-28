from util import load_data, perform_cross_validation, perform_cross_validation_oversampling, perform_test_report, \
    plot_importance_feature

TRAIN_DATA_LOC = './data/1631544228_021599_credit_train.csv'
TEST_DATA_LOC = './data/1631544228_0377958_credit_test.csv'

df = load_data(TRAIN_DATA_LOC, True)

X = df[df.columns[:-1]]
y = df[df.columns[-1]]

def main():

    exit_loop = False
    while not exit_loop:
        print("Please select a options")
        print("1. Cross Validation without oversampling")
        print("2. Cross Validation with oversampling")
        print("3. Perform data prediction on test set")
        print("4. Above all in sequence")
        print("5. Exit")
        choice = input("Enter your choice: ")

        if not choice.isnumeric():
            print("Oops, Please try again")
            continue

        choice = int(choice)

        if choice == 1:
            print("Cross Validation")
            perform_cross_validation(X, y)

        if choice == 2:
            print("Cross Validation with oversampling")
            perform_cross_validation_oversampling(X, y)

        if choice == 3:
            df_test = load_data(TEST_DATA_LOC)
            X_test = df_test[df_test.columns]
            perform_test_report(X, y, X_test)

        if choice == 4:
            print("Cross Validation")
            perform_cross_validation(X, y)

            print("Cross Validation with oversampling")
            perform_cross_validation_oversampling(X, y)

            df_test = load_data(TEST_DATA_LOC)
            X_test = df_test[df_test.columns]
            perform_test_report(X, y, X_test)

        elif choice == 5:
            exit_loop = True
        else:
            print("Oops, Please try again")
    return 0

main()
# plot_importance_feature(X, y)