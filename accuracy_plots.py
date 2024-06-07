import matplotlib.pyplot as plt
"""equal size results"""
# Data for the first experiment
sizes1 = [100, 200, 300, 400, 500]
test1 = [0.5, 0.5208333, 0.486111, 0.4916666, 0.5133333]
train1 = [0.5, 0.46875, 0.5208333, 0.5125, 0.48]

# Data for the second experiment
sizes2 = [100, 200, 300, 400, 500]
test2 = [0.8291666666666666, 0.85,
         0.8472222222222222, 0.1791666666666667, 0.8375]
train2 = [0.8361111111111111, 0.82222222, 0.8240740740740741,
          0.15833333333333333, 0.8305555555555555]
"""50 neg examples results"""
# # Data for the first experiment
# sizes1 = [100, 200, 300, 400, 500]
# test1 = [0.644444444, 0.806666666, 0.8619047619, 0.9037037037, 0.92121212]
# train1 = [0.7, 0.79, 0.85, 0.866666666, 0.89090909]

# # Data for the second experiment
# sizes2 = [100, 200, 300, 400, 500]
# test2 = [0.8909090909090909, 0.9523809523809523,
#          0.9629032258064516, 0.9707317073170731, 0.9784313725490196]
# train2 = [0.9212121212121213, 0.9523809523809523,
#           0.9709677419354839, 0.9788617886178862, 0.9816993464052287]


"""100 neg examples results"""
# # Data for the first experiment
# sizes1 = [100, 200, 300, 400, 500]
# test1 = [0.5, 0.35, 0.7375, 0.866666666, 0.836111]
# train1 = [0.5, 0.3083333, 0.76875, 0.79, 0.82916666]

# # Data for the second experiment
# sizes2 = [100, 200, 300, 400, 500]
# test2 = [0.8291666666666666, 0.925, 0.9328125, 0.9523809523809523, 0.9625]
# train2 = [0.8361111111111111, 0.8984848484848484,
#           0.940625, 0.9523809523809523, 0.960897435897435]

# Plot the results
plt.figure(figsize=(12, 6))

# Plot for the first experiment
plt.plot(sizes1, test1, 'o-', label='Federated - Test', color='blue')
plt.plot(sizes1, train1, 'o--', label='Federated - Train', color='blue')

# Plot for the second experiment
plt.plot(sizes2, test2, 's-', label='Non-federated - Test', color='red')
plt.plot(sizes2, train2, 's--', label='Non-federated - Train', color='red')

# Adding titles and labels
plt.title(
    'Federated vs. Non-federated Experiments Accuracy Given Same Size of Negative Examples')
plt.xlabel('Size of Positive Examples')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)

# Show the plot
plt.show()
