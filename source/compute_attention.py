from source.train_spatial_network import *
from vis.visualization.saliency import *
from vis.visualization.saliency import _find_penultimate_layer
from scipy.interpolate import interpn
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from source.data_visualization import *

modelPath = os.path.join(rootDir, 'data', 'model.bin')
weightPath = os.path.join(rootDir, 'data', 'model_weights.bin')
model = models.load_model(modelPath, {'auc_roc': auc_roc})
model.load_weights(weightPath)

print(K.image_data_format())

model.summary()

x_train, x_test, x_train_weights, _, y_train, y_test, ids_train, ids_test = get_final_model_dataset(True)


def visualize_cam_with_losses1(input_tensor, losses,
                               seed_input, penultimate_layer,
                               grad_modifier=None):
    penultimate_output = penultimate_layer.output
    opt = Optimizer(input_tensor, losses, wrt_tensor=penultimate_output, norm_grads=False)
    _, grads, penultimate_output_value = opt.minimize(seed_input, max_iter=10, grad_modifier=grad_modifier,
                                                      verbose=False)

    # For numerical stability. Very small grad values along with small penultimate_output_value can cause
    # w * penultimate_output_value to zero out, even for reasonable fp precision of float32.
    grads = grads / (np.max(grads) + K.epsilon())

    # Average pooling across all feature maps.
    # This captures the importance of feature map (channel) idx to the output.
    channel_idx = 1 if K.image_data_format() == 'channels_first' else -1
    other_axis = np.delete(np.arange(len(grads.shape)), channel_idx)
    weights = np.mean(grads, axis=tuple(other_axis))

    # Generate heatmap by computing weight * output over feature maps
    output_dims = utils.get_img_shape(penultimate_output)[2:]
    heatmap = np.zeros(shape=output_dims, dtype=K.floatx())
    for i, w in enumerate(weights):
        if channel_idx == -1:
            heatmap += w * penultimate_output_value[0, ..., i]
        else:
            heatmap += w * penultimate_output_value[0, i, ...]

    # ReLU thresholding to exclude pattern mismatch information (negative gradients).
    heatmap = np.maximum(heatmap, 0)

    # interpolate the heatmap to original size
    gridPoints = [np.arange(heatmap.shape[i]) for i in range(len(heatmap.shape))]

    eps = 1.0e-4
    inputShape = utils.get_img_shape(input_tensor)[2:]

    interpAxes = [np.linspace(0.0 + eps, heatmap.shape[i] - 1.0 - eps, inputShape[i]) for i in range(len(heatmap.shape))]

    inperpLocs = np.stack(np.meshgrid(*interpAxes, indexing='ij'), axis=3)

    linearInterpLoc = inperpLocs.reshape((-1, 3))

    newHeatmap = interpn(gridPoints, heatmap, linearInterpLoc).reshape(inputShape)
    newHeatmap = utils.normalize(newHeatmap)

    return newHeatmap


def visualize_cam1(model, layer_idx, filter_indices,
                  seed_input, penultimate_layer_idx=None,
                  backprop_modifier=None, grad_modifier=None):

    if backprop_modifier is not None:
        modifier_fn = get(backprop_modifier)
        model = modifier_fn(model)

    penultimate_layer = _find_penultimate_layer(model, layer_idx, penultimate_layer_idx)

    # `ActivationMaximization` outputs negative gradient values for increase in activations. Multiply with -1
    # so that positive gradients indicate increase instead.
    losses = [
        (ActivationMaximization(model.layers[layer_idx], filter_indices), -1)
    ]
    return visualize_cam_with_losses1(model.input, losses, seed_input, penultimate_layer, grad_modifier)


if __name__ == '__main__':
    y_test_labels = np.argmax(y_test, axis=1)

    testSample0s = np.where(y_test_labels == 0)[0]
    testSample1s = np.where(y_test_labels == 1)[0]

    result = visualize_cam1(model, -1, 1, x_test[testSample1s[4], ...])

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1, projection='3d')
    mappable = plot_activation_3d(ax, result, drawThreshold=0.6, cmapBase='jet', markerSize=140)
    plt.show()

    stats = [describe(np.take(newXs, ind, axis=4).flat) for ind in range(newXs.shape[4])]

    fig = plt.figure()
    axList = [fig.add_subplot(1, 5, i + 1, projection='3d') for i in range(5)]
    plot_affinity_3d(axList, x_test[testSample1s[4], ...], stats, drawThreshold=[0.6, 0.6, 0.6, 0.6, 0.6], markerSize=10)
    plt.show()

