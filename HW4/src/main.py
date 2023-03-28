from cross_validation import cross_validation
from save_graph import save_graph
from submissiob import main

while 1:
    print("Please select a options")
    print("1. Run Cross Validation")
    print("2. Run Submission")
    print("3. Run Both")
    print("4. Save Data Graph")
    print("5. Exit")
    choice = input("Enter your choice: ")

    if not choice.isnumeric():
       print("Oops, Please try again")
       continue

    choice = int(choice)

    if choice == 1:
       main()
    if choice == 2:
       cross_validation()
    if choice == 3:
       main()
       cross_validation()
    if choice == 4:
       save_graph()
    elif choice == 5:
        break