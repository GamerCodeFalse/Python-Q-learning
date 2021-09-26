import numpy as np

#actions codes: up = 0, down = 1,left = 2, right = 3 
actions = ["up","down","left","right"]

env_rows = 11
env_cols = 11

q_values = np.zeros((env_rows,env_cols,4))

rewards = np.full((env_rows,env_cols),-100.)
rewards[0,5] = 100

aisles = {}
aisles[1] = [i for i in range(1,10)]
aisles[2] = [1, 7, 9] 
aisles[3] = [i for i in range(1,8)]
aisles[3].append(9)
aisles[4] = [3, 7]  
aisles[5] = [i for i in range(11)]
aisles[6] = [5]  
aisles[7] = [i for i in range(1,10)]
aisles[8] = [3, 7]
aisles[9] = [i for i in range(11)]

for row_index in range(1,10):
    for collumn_index in aisles[row_index]:
        rewards[row_index,collumn_index] = -1

for row in rewards:
    print(row)

def is_terminal_state(current_row_index,current_collumn_index):
    if rewards[current_row_index,current_collumn_index] == -1:
        return False
    else:
        return True
    

def get_starting_location():
    current_row_index = np.random.randint(env_rows)
    current_collumn_index = np.random.randint(env_cols)

    while is_terminal_state(current_row_index, current_collumn_index):
        current_row_index = np.random.randint(env_rows)
        current_collumn_index = np.random.randint(env_cols)
    return current_row_index,current_collumn_index


def get_next_action(current_row_index,current_collumn_index,epsilon):
    if np.random.random() < epsilon:
        return np.argmax(q_values[current_row_index,current_collumn_index])
    else:
        return np.random.randint(4)


def get_next_location(current_row_index,current_collumn_index,action_index):
    new_row_index = current_row_index
    new_collumn_index = current_collumn_index
    if actions[action_index] == 'up' and current_row_index > 0:
        new_row_index -= 1
    elif actions[action_index] == 'down' and current_row_index < env_rows - 1:
        new_row_index += 1
    elif actions[action_index] == 'left' and current_collumn_index > 0:
        new_collumn_index -= 1
    elif actions[action_index] == 'right' and current_collumn_index < env_cols - 1:
        new_collumn_index += 1
    return new_row_index, new_collumn_index


def get_shortest_path(start_row_index,start_col_index):
    if is_terminal_state(start_row_index,start_col_index):
        return []
    else:
        current_row_index, current_collumn_index = start_row_index, start_col_index
        shortest_path = []
        shortest_path.append([current_row_index,current_collumn_index])
        while not is_terminal_state(current_row_index,current_collumn_index):
            action_index = get_next_action(current_row_index,current_collumn_index,1.)
            current_row_index,current_collumn_index = get_next_location(current_row_index,current_collumn_index,action_index)
            shortest_path.append([current_row_index,current_collumn_index])
        return shortest_path

epsilon = 0.9
discount_factor = 0.9
learning_rate = 0.9

for episode in range(1000):
    row_index,collumn_index = get_starting_location()

    while not is_terminal_state(row_index,collumn_index):
        action_index = get_next_action(row_index,collumn_index,epsilon)
        old_row_index,old_collumn_index = row_index,collumn_index 
        row_index,collumn_index = get_next_location(row_index,collumn_index,action_index)
        reward  = rewards[row_index,collumn_index]
        old_q_value = q_values[old_row_index,old_collumn_index,action_index]
        temporal_differnce = reward + (discount_factor*np.max(q_values[row_index,collumn_index]))-old_q_value
        new_q_value = old_q_value + (learning_rate*temporal_differnce)
        q_values[old_row_index,old_collumn_index,action_index] = new_q_value
print("Training Complete")
print(get_shortest_path(5,0))