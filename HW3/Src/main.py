from part1 import run_part_1
from part2 import run_part_2

while 1:
    print("Please select a options")
    print("1. Part 1 K-Means")
    print("2. Part 1 K-Means Bisect")
    print("3. Part 1 Plot data graph")
    print("4. Part 2 K-Means")
    print("5. Part 2 K-Means Bisect")
    print("6. Part 2 plot K vs SSEs")
    print("7. Exit")
    choice = input("Enter your choice: ")

    if not choice.isnumeric():
        print("Oops, Please try again")
        continue

    choice = int(choice)

    if choice == 1 or choice == 2 or choice == 3:
        run_part_1(choice)
    elif choice == 4 or choice == 5 or choice == 6:
        run_part_2(choice - 3)
    elif choice == 7:
        break
