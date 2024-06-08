import matplotlib.pyplot as plt
"""equal size results"""
# # Data for the first experiment
# sizes1 = [100, 200, 300, 400, 500]
# test1 = [0.6599999999999999, 0.6725, 0.68, 0.6875, 0.694]
# train1 = [0.6762499999999999, 0.6918749999999999,
#           0.7012500000000002, 0.7174999999999999, 0.71425]

# # Data for the second experiment
# sizes2 = [100, 200, 300, 400, 500]
# test2 = [0.8291666666666666, 0.85,
#          0.8472222222222222, 0.8472222222222222, 0.8375]
# train2 = [0.8361111111111111, 0.82222222, 0.8240740740740741,
#           0.8240740740740741, 0.8305555555555555]
"""50 neg examples results"""
# # Data for the first experiment
# sizes1 = [100, 200, 300, 400, 500]
# test1 = [0.7266666666666667, 0.6519999999999999,
#          0.6657142857142857, 0.6422222222222222, 0.6763636363636364]
# train1 = [0.7183333333333334, 0.6559999999999999,
#           0.6578571428571429, 0.6688888888888889, 0.6295454545454544]

# # Data for the second experiment
# sizes2 = [100, 200, 300, 400, 500]
# test2 = [0.8909090909090909, 0.9523809523809523,
#          0.9629032258064516, 0.9707317073170731, 0.9784313725490196]
# train2 = [0.9212121212121213, 0.9523809523809523,
#           0.9709677419354839, 0.9788617886178862, 0.9816993464052287]


"""100 neg examples results"""
# Data for the first experiment
sizes1 = [100, 200, 300, 400, 500]
test1 = [0.6599999999999999, 0.64, 0.67,
         0.7060000000000001, 0.7116666666666667]
train1 = [0.6762499999999999, 0.6916666666666667,
          0.690625, 0.6940000000000001, 0.6779166666666667]

# Data for the second experiment
sizes2 = [100, 200, 300, 400, 500]
test2 = [0.8291666666666666, 0.925, 0.9328125, 0.9523809523809523, 0.9625]
train2 = [0.8361111111111111, 0.8984848484848484,
          0.940625, 0.9523809523809523, 0.960897435897435]

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
    'Federated vs. Non-federated Experiments Accuracy Given 100 Negative Examples')
plt.xlabel('Size of Positive Examples')
plt.ylabel('Accuracy')
plt.ylim(0, 1)
plt.legend()
plt.grid(True)

# Show the plot
plt.show()
