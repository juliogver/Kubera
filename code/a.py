import random

def check_enzo_bodycount():
    enzo_bodycount = random.randint(0, 5)  # Simulate Enzo's bodycount (change logic if needed)
    print(f"Enzo's bodycount: {enzo_bodycount}")

    if enzo_bodycount >= 3:
        return True
    else:
        return check_enzo_bodycount()  # Recursively rerun the function

result = check_enzo_bodycount()
print("Final result:", result)