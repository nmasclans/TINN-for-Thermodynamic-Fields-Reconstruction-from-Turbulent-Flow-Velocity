import numpy as np
from matplotlib import pyplot as plt
from matplotlib import colors
import tensorflow as tf

from my_utils import transform_targets_to_original_scaling, classify_state_from_temperature
from my_metrics_classification import compute_metrics

# colormap classification
blue_0  = [0.0,150/255,1.0,0.74]
green_1 = [0.0,150/255,0.0,0.57]
red_2   = [1.0,0.0,0.0,0.54]
list_colors = np.array([blue_0, green_1, red_2])
cmap = colors.ListedColormap(list_colors)
bounds = [-0.5,0.5,1.5,2.5]
norm = colors.BoundaryNorm(bounds,cmap.N)

# Plot the quantities of the validation dataset, single validation batch
# Plot scatterplot + histogram

def visualize_prediction(y_gt, y_pred, epoch, batch, args):
    (y_gt, y_pred) = transform_targets_to_original_scaling(args, y_gt, y_pred)
    for target_idx in range(args.num_targets):
        y_gt_0   = y_gt[:,target_idx]
        y_pred_0 = y_pred[:,target_idx]
        target_name = args.targets_name[target_idx]
        # probability distribution
        fig_title = f"figures/{target_name}_kde_E{epoch}_B{batch}.png"
        plt.figure()
        plt.hist(x=y_gt_0,  bins=100,label="ground-truth")
        plt.hist(x=y_pred_0,bins=100,label="prediction")
        plt.ylabel(f"{target_name} kde, density of probability")
        plt.legend()
        plt.savefig(fig_title)
        plt.close()
        # print(f"Visualization of validation results in '{fig_title}'")
        # scatter plot
        fig_title = f"figures/{target_name}_scatterplot_E{epoch}_B{batch}.png"
        plt.figure()
        plt.scatter(x=y_gt_0[::args.visualization_step],y=y_pred_0[::args.visualization_step],s=1)
        plt.xlabel(f'scaled {target_name} (ground truth)')
        plt.ylabel(f'scaled {target_name} (prediction)')
        plt.savefig(fig_title)
        plt.close()
        # print(f"Visualization of validation results in '{fig_title}'")

def visualize_prediction_regression_by_xyplanes(y_gt, y_pred, epoch, batch, args):
    assert y_gt.ndim == 2 and y_pred.ndim == 2
    assert y_gt.shape[0] == 128**3 and y_pred.shape[0] == 128**3, "--batch_size of prediction must be set to 128**3"
    (y_gt, y_pred) = transform_targets_to_original_scaling(args, y_gt, y_pred)
    for target_idx in range(args.num_targets):
        y_gt_target   = y_gt[:,target_idx]
        y_pred_target = y_pred[:,target_idx]
        target_name = args.targets_name[target_idx]
        # recover spatial dimension
        y_gt_target_d   = tf.reshape(y_gt_target,   args.spatial_dimension)         # shape: [128,128,128]
        y_pred_target_d = tf.reshape(y_pred_target, args.spatial_dimension)
        # plot imshow of middle plane z
        # ground truth
        fig_title = f"figures/{target_name}_contourf_E{epoch}_B{batch}_gt.png"
        fig_title_2 = f"figures/{target_name}_contourf_E{epoch}_B{batch}_gt.svg"
        plt.figure()
        plt.imshow(y_gt_target_d[64,:,:],cmap='coolwarm'); plt.axis('scaled'); plt.colorbar()
        frame = plt.gca()
        frame.invert_yaxis()
        plt.xticks([]),plt.yticks([])
        plt.savefig(fig_title); plt.savefig(fig_title_2)
        plt.close()
        # prediction
        fig_title = f"figures/{target_name}_contourf_E{epoch}_B{batch}_pred.png"
        fig_title_2 = f"figures/{target_name}_contourf_E{epoch}_B{batch}_pred.svg"
        plt.figure()
        plt.imshow(y_pred_target_d[64,:,:],cmap='coolwarm'); plt.axis('scaled'); plt.colorbar()
        frame = plt.gca()
        frame.invert_yaxis()
        plt.xticks([]),plt.yticks([])
        plt.savefig(fig_title); plt.savefig(fig_title_2)
        plt.close()

def visualize_prediction_classification_by_xyplanes(y_gt, y_pred, epoch, batch, args):
    assert y_gt.ndim == 2 and y_pred.ndim == 2
    assert y_gt.shape[0] == 128**3 and y_pred.shape[0] == 128**3, "--batch_size of prediction must be set to 128**3"
    assert args.targets_name[2] == 'T', "'visualize_prediction_classification_by_xyplanes' function needs --target_name 3rd element to be 'T'"
    (y_gt, y_pred) = transform_targets_to_original_scaling(args, y_gt, y_pred)
    T_gt   = y_gt[:,2]
    T_pred = y_pred[:,2]
    # predict class from regression data!
    (state_gt, state_pred) = classify_state_from_temperature(args, T_gt, T_pred)
    # compute (and print) metrics
    compute_metrics(state_gt, state_pred)
    # recover spatial dimension
    state_gt_d   = tf.reshape(state_gt,   args.spatial_dimension)   # shape: [128,128,128]
    state_pred_d = tf.reshape(state_pred, args.spatial_dimension)
    # plot contourf of middle plane z
    # ground truth
    fig_title = f"figures/fluid_state_contourf_E{epoch}_B{batch}_gt.png"
    fig_title_2 = f"figures/fluid_state_contourf_E{epoch}_B{batch}_gt.svg"
    plt.figure()
    img = plt.imshow(state_gt_d[64,:,:], interpolation='nearest', origin='lower',
               cmap=cmap, norm=norm)
    plt.gca().invert_yaxis()
    plt.axis('scaled')
    plt.colorbar(img, cmap=cmap, norm=norm, boundaries=bounds)
    plt.xticks([]),plt.yticks([])
    plt.savefig(fig_title); plt.savefig(fig_title_2)
    plt.close()
    # prediction
    fig_title = f"figures/fluid_state_contourf_E{epoch}_B{batch}_pred.png"
    fig_title_2 = f"figures/fluid_state_contourf_E{epoch}_B{batch}_pred.svg"
    plt.figure()
    img = plt.imshow(state_pred_d[64,:,:], interpolation='nearest', origin='lower',
               cmap=cmap, norm=norm)
    plt.gca().invert_yaxis()
    plt.axis('scaled')
    plt.colorbar(img, cmap=cmap, norm=norm, boundaries=bounds)
    plt.xticks([]),plt.yticks([])
    plt.savefig(fig_title); plt.savefig(fig_title_2)
    plt.close()
