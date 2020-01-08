import numpy as np

from CNN.layers import convolutional_layer, maxpool_layer, loss_function, softmax, backpropagation_maxpool, backpropagation_convolutionalLayer


def build_net(image, label, parameters, stride, pooling_filter, pooling_stride):
    [weight1, weight2, weight3, weight4, bias1, bias2, bias3, bias4] = parameters

    #forward propagation
    first_convolution = convolutional_layer(image, weight1, bias1, stride)
    first_convolution[first_convolution <= 0] = 0

    second_convolution = convolutional_layer(first_convolution, weight2, bias2, stride)
    second_convolution[second_convolution <= 0] = 0

    pooling_layer = maxpool_layer(second_convolution, pooling_filter, pooling_stride)

    (num_filters, height_width, same) = pooling_layer.shape

    flatten = pooling_layer.reshape((num_filters * height_width * height_width, 1))

    fully_connected1 = weight3.dot(flatten) + bias3
    fully_connected1[fully_connected1 <= 0] = 0

    fully_connected2 = weight4.dot(fully_connected1) + bias4

    prediction = softmax(fully_connected2)

    loss = loss_function(prediction, label)

    #backpropagation
    derivative_second_fully_conn = prediction - label
    gradient_weight4 = derivative_second_fully_conn.dot(fully_connected1.T)
    gradient_bias4 = np.sum(derivative_second_fully_conn, axis=1).reshape(bias4.shape)

    derivative_first_fully_conn = weight4.T.dot(derivative_second_fully_conn)
    derivative_first_fully_conn[fully_connected1 <= 0] = 0
    gradient_weight3 = derivative_first_fully_conn.dot(flatten.T)
    gradient_bias3 = np.sum(derivative_first_fully_conn, axis=1).reshape(bias3.shape)

    der_second_fc = weight3.T.dot(derivative_first_fully_conn)
    der_maxpool = der_second_fc.reshape(pooling_layer.shape)

    der_conv2 = backpropagation_maxpool(der_maxpool, second_convolution, pooling_filter, pooling_stride)
    der_conv2[second_convolution <= 0] = 0

    der_conv1, gradient_weight2, der_bias2 = backpropagation_convolutionalLayer(der_conv2, first_convolution, weight2, stride)
    der_conv1[first_convolution <= 0] = 0

    image_der, gradient_weight1, der_bias1 = backpropagation_convolutionalLayer(der_conv1, image, weight1, stride)

    gradients = [gradient_weight1, gradient_weight2, gradient_weight3, gradient_weight4, bias1, bias2, bias3, bias4]
    return gradients, loss


# Adams optimizer

def adam_optimizer(batch, num_classes, alpha, dim, n_c, beta1, beta2, parameters, cost_array, E=1e-7):

    [weight1, weight2, weight3, weight4, bias1, bias2, bias3, bias4] = parameters

    batch_size = len(batch)

    images = batch[:, 0:-1]
    images = images.reshape((batch_size, n_c, dim, dim))

    labels = batch[:, -1]

    cost = 0

    # initialize gradients with zeros
    grad_w1 = np.zeros(weight1.shape)
    grad_w2 = np.zeros(weight2.shape)
    grad_w3 = np.zeros(weight3.shape)
    grad_w4 = np.zeros(weight4.shape)
    grad_b1 = np.zeros(bias1.shape)
    grad_b2 = np.zeros(bias2.shape)
    grad_b3 = np.zeros(bias3.shape)
    grad_b4 = np.zeros(bias4.shape)

    # initialize momentum parameters with zeros
    moment_param_w1 = np.zeros(weight1.shape)
    moment_param_w2 = np.zeros(weight2.shape)
    moment_param_w3 = np.zeros(weight3.shape)
    moment_param_w4 = np.zeros(weight4.shape)
    moment_param_b1 = np.zeros(bias1.shape)
    moment_param_b2 = np.zeros(bias2.shape)
    moment_param_b3 = np.zeros(bias3.shape)
    moment_param_b4 = np.zeros(bias4.shape)

    # initialize RMS-prop parameters with zeros
    rmsprop_w1 = np.zeros(weight1.shape)
    rmsprop_w2 = np.zeros(weight2.shape)
    rmsprop_w3 = np.zeros(weight3.shape)
    rmsprop_w4 = np.zeros(weight4.shape)
    rmsprop_b1 = np.zeros(bias1.shape)
    rmsprop_b2 = np.zeros(bias2.shape)
    rmsprop_b3 = np.zeros(bias3.shape)
    rmsprop_b4 = np.zeros(bias4.shape)


    for i in range(batch_size):
        image = images[i]
        label = np.eye(num_classes)[int(labels[i])].reshape((num_classes, 1))

        gradients, loss = build_net(image, label, parameters, 1, 2, 2)

        [gradient_weight1, gradient_weight2, gradient_weight3, gradient_weight4, bias1, bias2, bias3, bias4] = gradients

        grad_w1 += gradient_weight1
        grad_w2 += gradient_weight2
        grad_w3 += gradient_weight3
        grad_w4 += gradient_weight4
        grad_b1 += bias1
        grad_b2 += bias2
        grad_b3 += bias3
        grad_b4 += bias4

        cost += loss

    # update momentum and RMS-prop parameters
    moment_param_w1 = beta1 * moment_param_w1 + (1 - beta1) * grad_w1 / batch_size
    rmsprop_w1 = beta2 * rmsprop_w1 + (1 - beta2) * (grad_w1 / batch_size) ** 2
    weight1 -= alpha * moment_param_w1 / np.sqrt(rmsprop_w1 + E)

    moment_param_w2 = beta1 * moment_param_w2 + (1 - beta1) * grad_w2 / batch_size
    rmsprop_w2 = beta2 * rmsprop_w2 + (1 - beta2) * (grad_w2 / batch_size) ** 2
    weight2 -= alpha * moment_param_w2 / np.sqrt(rmsprop_w2 + E)

    moment_param_w3 = beta1 * moment_param_w3 + (1 - beta1) * grad_w3 / batch_size
    rmsprop_w3 = beta2 * rmsprop_w3 + (1 - beta2) * (grad_w3 / batch_size) ** 2
    weight3 -= alpha * moment_param_w3 / np.sqrt(rmsprop_w3 + E)

    moment_param_w4 = beta1 * moment_param_w4 + (1 - beta1) * grad_w4 / batch_size
    rmsprop_w4 = beta2 * rmsprop_w4 + (1 - beta2) * (grad_w4 / batch_size) ** 2
    weight4 -= alpha * moment_param_w4 / np.sqrt(rmsprop_w4 + E)

    moment_param_b1 = beta1 * moment_param_b1 + (1 - beta1) * grad_b1 / batch_size
    rmsprop_b1 = beta2 * rmsprop_b1 + (1 - beta2) * (grad_b1 / batch_size) ** 2
    bias1 -= alpha * moment_param_b1 / np.sqrt(rmsprop_b1 + E)

    moment_param_b2 = beta1 * moment_param_b2 + (1 - beta1) * grad_b2 / batch_size
    rmsprop_b2 = beta2 * rmsprop_b2 + (1 - beta2) * (grad_b2 / batch_size) ** 2
    bias2 -= alpha * moment_param_b2 / np.sqrt(rmsprop_b2 + E)

    moment_param_b3 = beta1 * moment_param_b3 + (1 - beta1) * grad_b3 / batch_size
    rmsprop_b3 = beta2 * rmsprop_b3 + (1 - beta2) * (grad_b3 / batch_size) ** 2
    bias3 -= alpha * moment_param_b3 / np.sqrt(rmsprop_b3 + E)

    moment_param_b4 = beta1 * moment_param_b4 + (1 - beta1) * grad_b4 / batch_size
    rmsprop_b4 = beta2 * rmsprop_b4 + (1 - beta2) * (grad_b4 / batch_size) ** 2
    bias4 -= alpha * moment_param_b4 / np.sqrt(rmsprop_b4 + E)

    cost = cost / batch_size
    cost_array.append(cost)

    parameters = [weight1, weight2, weight3, weight4, bias1, bias2, bias3, bias4]

    return parameters, cost_array