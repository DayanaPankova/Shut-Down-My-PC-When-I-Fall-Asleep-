import numpy as np

def convolutional_layer(image, filter, bias, stride=1):

    (num_channels, img_h, img_w) = image.shape
    (num_filters, num_channels, filter_height_width, filter_height_width) = filter.shape

    output_height = int((img_h - filter_height_width) / stride) + 1
    output_width = int((img_h - filter_height_width) / stride) + 1

    output = np.zeros((num_filters, output_height, output_width))

    for i in range(num_filters):
        row = out_row = 0

        while row + filter_height_width <= img_h:

            column = out_column = 0

            while column + filter_height_width <= img_w:
                output[i, out_row, out_column] = np.sum(filter[i] * image[:, row: row + filter_height_width, column: column + filter_height_width]) + bias[i]
                column += stride
                out_column += 1

            row += stride
            out_row += 1

    return output


def maxpool_layer(image, filter=5, stride=2):
    (channels, img_height, img_width) = image.shape
    output_img_height = int((img_height - filter) / stride) + 1
    output_img_width = int((img_width - filter) / stride) + 1
    img = np.zeros((channels, output_img_height, output_img_width))

    for i in range(channels):
        row = out_row = 0

        while row + filter <= img_height:

            column = out_column = 0

            while column + filter <= img_width:
                img[i, out_row, out_column] = np.max(image[i, row: row + filter, column: column + filter])
                column += stride
                out_column += 1

            row += stride
            out_row += 1

    return img


def softmax(input):
    raise_power_e = np.exp(input)
    probabilities = raise_power_e / np.sum(raise_power_e)
    return probabilities


def loss_function(prediction, label):
    net_loss = -np.sum(label * np.log(prediction))   #cross-entropy formula
    return net_loss


def backpropagation_convolutionalLayer(derivativeFromPreviousLayer, convLayer, filter, stride):
    (num_filters, num_dims, filter_height_width, filter_height_width) = filter.shape
    (num_dims, img_height, img_width) = convLayer.shape

    derived_image = np.zeros(convLayer.shape)
    derived_filter = np.zeros(filter.shape)
    derived_bias = np.zeros((num_filters, 1))
    for i in range(num_filters):
        row = der_img_height = 0
        while row + filter_height_width <= img_height:
            col = der_img_width = 0
            while col + filter_height_width <= img_width:
                derived_filter[i] += derivativeFromPreviousLayer[i, der_img_height, der_img_width] * convLayer[:, row:row + filter_height_width, col:col + filter_height_width]
                derived_image[:, row:row + filter_height_width, col:col + filter_height_width] += derivativeFromPreviousLayer[i, der_img_height, der_img_width] * filter[i]
                col += stride
                der_img_width += 1
            row += stride
            der_img_height += 1
        derived_bias[i] = np.sum(derivativeFromPreviousLayer[i])
    return derived_image, derived_filter, derived_bias



def backpropagation_maxpool(pool_der, maxpooled, filter, stride):
    (num_dims, img, ok) = maxpooled.shape
    maxpool_der = np.zeros(maxpooled.shape)

    for i in range(num_dims):
        row = height = 0
        while row + filter <= img:
            col = width = 0
            while col + filter <= img:
                index = np.nanargmax(maxpooled[i, row:row + filter, col:col + filter])
                (a, b) = np.unravel_index(index, maxpooled[i, row:row + filter, col:col + filter].shape)
                maxpool_der[i, row + a, col + b] = pool_der[i, height, width]

                col += stride
                width += 1
            row += stride
            height += 1

    return maxpool_der


def predict(image, parameters, stride = 1, pool_filter = 2, pool_stride = 2):
    [f1, f2, w3, w4, b1, b2, b3, b4] = parameters

    first_convolution = convolutional_layer(image, f1, b1, stride)
    first_convolution[first_convolution <= 0] = 0  #relu

    second_convolution = convolutional_layer(first_convolution, f2, b2, stride)
    second_convolution[second_convolution <= 0] = 0  #relu

    pooling_layer = maxpool_layer(second_convolution, pool_filter, pool_stride)

    (num_filters, h_w, ok) = pooling_layer.shape
    fc = pooling_layer.reshape((num_filters * h_w * h_w, 1))  # flatten
    fully_connected1 = w3.dot(fc) + b3

    fully_connected1[fully_connected1 <= 0] = 0  #relu

    fully_connected2 = w4.dot(fully_connected1) + b4

    probabilities = softmax(fully_connected2)

    prediction = np.argmax(probabilities)
    probability = np.max(probabilities)

    return prediction, probability
