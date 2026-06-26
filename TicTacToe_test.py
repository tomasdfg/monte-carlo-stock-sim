# setting board 
x1 = 0
x2 = 0
x3 = 0
y1 = 0
y2 = 0
y3 = 0
z1 = 0
z2 = 0
z3 = 0
spaces = [x1, x2, x3, y1, y2, y3, z1, z2, z3]
#user_want_to_play = input("do you want to play? Y/N: ")
#if user_want_to_play == "N":
    #pass
#elif user_want_to_play == "Y":
while (int(x1) + int(x2) + int(x3) + int(y1) + int(y2) + int(y3) + int(z1) + int(z2) + int(z3)) != int(9):
    user_choice_1 = input("Where would you like to go: ")
    if spaces[user_choice_1.index.__str__] == 0:
           spaces[user_choice_1.index] += 1


    if user_choice_1 == "1":
            if x1 == 0:
                x1 += 1
            else:
                print("Cant choose that one")
    elif user_choice_1 == "2":
            x2 += 1
    elif user_choice_1 == "3":
            x3 += 1
    elif user_choice_1 == "4":
            y1 += 1
    elif user_choice_1 == "5":
            y2 += 1
    elif user_choice_1 == "6":
            y3 += 1
    elif user_choice_1 == "7":
            z1 += 1
    elif user_choice_1 == "8":
            z2 += 1
    elif user_choice_1 == "9":
            z3 += 1
        
else:
        print("board filled")
    

    