# Operators
def move(subject, x1, x2):
    return f"Move {subject} from {x1} to {x2}"

def push_box(x1, x2):
    return f"Push box from {x1} to {x2}"

def climb_box(x, direction):
    return f"Climb box at {x} {direction}"

def have_banana(x):
    return f"Have banana at {x}"


# Initial State
initial_state = {
    'monkeyAt0': True,
    'monkeyLevel': 'Down',
    'bananaAt1': True,
    'boxAt2': True
}

# Goal State
goal_state = {
    'GetBanana': True,
    'at': 1
}

# Planning Algorithm
def plan_actions(initial_state, goal_state):
    actions = []
    # Example planning algorithm to achieve the goal state
    if initial_state['monkeyAt0'] and initial_state['bananaAt1']:
        actions.append(move('Monkey', 0, 1))
        actions.append(climb_box(1, 'Up'))
        actions.append(have_banana(1))
    return actions


# Execute the planning algorithm
actions = plan_actions(initial_state, goal_state)

# Print the actions in the plan
print("Plan:")
for action in actions:
    print(action)
