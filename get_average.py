import numpy as np

# Define the data
data = {
    100: {
        "train": [0.61875, 0.7625, 0.8375, 0.64375, 0.51875],
        "test": [0.675, 0.675, 0.825, 0.575, 0.55]
    },
    200: {
        "train": [0.6916666666666667, 0.7333333333333334, 0.7, 0.7333333333333334, 0.6],
        "test": [0.5666666666666667, 0.6666666666666667, 0.7833333333333333, 0.7, 0.4833333333333333]
    },
    300: {
        "train": [0.678125, 0.684375, 0.703125, 0.75625, 0.63125],
        "test": [0.6125, 0.7375, 0.7125, 0.725, 0.5625]
    },
    400: {
        "train": [0.6875, 0.69, 0.7050000000000001, 0.7475, 0.64],
        "test": [0.65, 0.72, 0.75, 0.77, 0.64]
    },
    500: {
        "train": [0.675, 0.6895833333333333, 0.7083333333333333, 0.7333333333333334, 0.5833333333333333],
        "test": [0.725, 0.7416666666666667, 0.7, 0.75, 0.6416666666666666]
    }
}

# Function to calculate average accuracies


def calculate_averages(data):
    averages = {}
    for size, results in data.items():
        train_avg = np.mean(results['train'])
        test_avg = np.mean(results['test'])
        averages[size] = {
            'train_avg': train_avg,
            'test_avg': test_avg
        }
    return averages


# Calculate the averages
averages = calculate_averages(data)

tests = []
trains = []
# Print the results
for size, avg in averages.items():
    trains.append(avg['train_avg'])
    tests.append(avg['test_avg'])

print(f"test1={tests}")
print(f"train1={trains}")
