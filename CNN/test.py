from CNN.functions import *
from CNN.layers import predict
from tqdm import tqdm
import pickle as pickle

model = 'parameters.pkl'
parameters, cost = pickle.load(open(model, 'rb'))
[weight1, weight2, weight3, weight4, bias1, bias2, bias3, bias4] = parameters
#get data
num_images = 10000
X = extract_images('test_images.gz', num_images, 32)
y_dash = extract_labels('test_labels.gz', num_images).reshape(num_images, 1)
#normalize
X -= int(np.mean(X))
X /= int(np.std(X))
test_data = np.hstack((X, y_dash))

X = test_data[:, 0:-1]
X = X.reshape(len(test_data), 1, 32, 32)
y = test_data[:, -1]

correct = 0
digit_count = [0 for i in range(2)]
digit_correct = [0 for i in range(2)]

print()
print("Computing accuracy...")

t = tqdm(range(len(X)), leave=True)

for i in t:
    x = X[i]
    pred, prob = predict(x, parameters)
    digit_count[int(y[i])] += 1
    if pred == y[i]:
        correct += 1
        digit_correct[pred] += 1

    t.set_description("Accuracy so far: %0.2f%%" % (float(correct / (i + 1)) * 100))

print("Overall Accuracy: %.2f" % (float(correct / len(test_data) * 100)))
