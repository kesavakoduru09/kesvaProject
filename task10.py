# Define a list of facts
facts = [
    "john_is_cold.",
    "raining.",
    # john is cold
    # it is raining
    "john_Forgot_His_Raincoat.",  # john forgot his raincoat
    "fred_lost_his_car_keys.",
    # fred lost his car keys
    "peter_footballer."
]
# peter plays football

# Function to check if a fact is true
def verify_fact(fact):
    # Remove the trailing period
    fact = fact.rstrip(".")
    # Perform some logic to verify the fact
    if fact == "john_Forgot_His_Raincoat":
        return True
    elif fact == "raining":
        return True
    elif fact == "foggy":
        return True
    elif fact == "Cloudy":
        return False  # Assume it's not cloudy
    else:
        return False

# Verify each fact
for fact in facts:
    if verify_fact(fact):
        print(f"{fact} - Yes")
    else:
        print(f"{fact} - No")
