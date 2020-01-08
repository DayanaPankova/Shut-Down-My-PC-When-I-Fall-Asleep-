from CNN.functions import *
from CNN.build_net import *
import matplotlib.pyplot as plt
from tqdm import tqdm
import pickle as pickle


def train(num_possible_outcomes=2, alpha=0.01, beta1=0.95, beta2=0.99, image_dimensions=32, img_depth=1,
          filter_size=5, num_filters_first=8, num_filters_second=8, batch_size=56, num_epochs=10, model='newparameters.pkl'):
    num_imgs = 56000
    X = extract_images('train_images.gz', num_imgs, image_dimensions)
    y = extract_labels('train_labels.gz', num_imgs).reshape(num_imgs, 1)

    X -= int(np.mean(X))
    X /= int(np.std(X))
    train_data = np.hstack((X, y))
    np.random.shuffle(train_data)

    weight1, weight2, weight3, weight4 = (num_filters_first, img_depth, filter_size, filter_size), (num_filters_second, num_filters_first, filter_size, filter_size), (128, 1152), (2, 128)
    weight1 = convLayer_weights(weight1)
    weight2 = convLayer_weights(weight2)
    weight3 = fullyConnected_weights(weight3)
    weight4 = fullyConnected_weights(weight4)

    bias1 = np.zeros((weight1.shape[0], 1))
    bias2 = np.zeros((weight2.shape[0], 1))
    bias3 = np.zeros((weight3.shape[0], 1))
    bias4 = np.zeros((weight4.shape[0], 1))

    parameters = [weight1, weight2, weight3, weight4, bias1, bias2, bias3, bias4]

    cost_array = []

    for epoch in range(num_epochs):
        print("Training for epoch " + str(epoch + 1))
        print("Deployed epoch " + str(epoch + 1))
        np.random.shuffle(train_data)
        batches = [train_data[k:k + batch_size] for k in range(0, train_data.shape[0], batch_size)]

        t = tqdm(batches)
        for x, batch in enumerate(t):
            parameters, cost_array = adam_optimizer(batch, num_possible_outcomes, alpha, image_dimensions, img_depth, beta1, beta2, parameters, cost_array, E=1e-7)
            t.set_description("Cost: %.2f" % (cost_array[-1]))

    save = [parameters, cost_array]
    with open(model, 'wb') as file:
        pickle.dump(save, file)

    return cost_array


cost = train()

print("Network trained succesfully")
parameters, cost = pickle.load(open('newparameters.pkl', 'rb'))
[f1, f2, w3, w4, b1, b2, b3, b4] = parameters

# Plot cost
plt.plot(cost, 'b')
plt.xlabel('# Iterations')
plt.ylabel('Cost')
plt.legend('Loss', loc='upper right')
plt.show()