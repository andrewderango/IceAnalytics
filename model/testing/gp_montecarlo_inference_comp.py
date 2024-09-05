import numpy as np

def simulate_probability(values, target_value, trials=100000):
    total_picks = 12  # Number of values to pick in each trial
    count_selected = 0
    
    for _ in range(trials):
        # Pick 12 values without replacement, weighted by the values in the list
        selected = np.random.choice(values, total_picks, replace=False, p=np.array(values)/np.sum(values))
        
        # Check if the target value is in the selected list
        if target_value in selected:
            count_selected += 1
    
    # Calculate the probability
    probability = count_selected / trials
    return probability

# Example usage:
values = [0.990440044, 0.9866352065615764, 0.93844059900821, 0.8800997582857142, 0.9156115743776684, 0.6987885723096641, 0.7737041550722896, 0.7785463099843806, 0.6011144594497176, 0.0940200035243702, 0.4665413307301053, 0.4368370627578196, 0.4232441837478473, 0.2950138804918099, 0.067481563899235, 0.0106247751211502, 0.0905899762906003]
target_value = 0.990440044
probability = simulate_probability(values, target_value)
print(f"Estimated probability that {target_value} is selected: {probability}")