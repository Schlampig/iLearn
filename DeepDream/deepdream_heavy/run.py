# references: https://github.com/gordicaleksa/pytorch-deepdream
from utils import *
from copy import deepcopy
import argparse
import numpy as np
import cv2 as cv
import torch


# Configs
####################################################################################################
parser = argparse.ArgumentParser()
parser.add_argument("--img_width", type=int, default=600)
parser.add_argument("--layers_to_use", type=str, nargs='+', default=['relu4_3'])
parser.add_argument("--model_name", help="Used network name for DeepDream.", default='VGG16')
parser.add_argument("--pretrained_weights", help="Pretrained weights for the used network.", default='IMAGENET')
parser.add_argument("--pyramid_size", type=int, default=4)
parser.add_argument("--pyramid_ratio", type=float, default=1.8)
parser.add_argument("--alpha", help="Ratio to mix-up new and old images.", type=float, default=0.5)
parser.add_argument("--num_gradient_ascent_iterations", type=int, default=5)
parser.add_argument("--lr", type=float, default=0.09)
parser.add_argument("--should_display", type=bool, default=False)
parser.add_argument("--spatial_shift_size", type=int, default=32)
parser.add_argument("--smoothing_coefficient", type=float, default=0.5)
parser.add_argument("--use_noise", type=bool, default=False)
args = parser.parse_args()
config = dict()
for arg in vars(args):
    config[arg] = getattr(args, arg)
config['dump_dir'] = "./dump_dir"
config['input'] = "figures.jpg"


# DeepDream Core
####################################################################################################
def gradient_ascent(config, model, input_tensor, layer_ids_to_use, iteration):
    # Step 0: Feed forward pass
    out = model(input_tensor)

    # Step 1: Grab activations/feature maps of interest
    activations = [out[layer_id_to_use] for layer_id_to_use in layer_ids_to_use]

    # Step 2: Calculate loss over activations
    losses = []
    for layer_activation in activations:
        loss_component = torch.nn.MSELoss(reduction='mean')(layer_activation, torch.zeros_like(layer_activation))
        losses.append(loss_component)
    loss = torch.mean(torch.stack(losses))
    loss.backward()

    # Step 3: Process image gradients (smoothing + normalization, more an art then a science)
    grad = input_tensor.grad.data
    sigma = ((iteration + 1) / config['num_gradient_ascent_iterations']) * 2.0 + config['smoothing_coefficient']
    smooth_grad = CascadeGaussianSmoothing(kernel_size=9, sigma=sigma)(grad)  # "magic number" 9 just works well
    g_std = torch.std(smooth_grad)
    g_mean = torch.mean(smooth_grad)
    smooth_grad = smooth_grad - g_mean
    smooth_grad = smooth_grad / g_std

    # Step 4: Update image using the calculated gradients (gradient ascent step)
    input_tensor.data += config['lr'] * smooth_grad

    # Step 5: Clear gradients and clamp the data (otherwise values would explode to +- "infinity")
    input_tensor.grad.data.zero_()
    input_tensor.data = torch.max(torch.min(input_tensor, UPPER_IMAGE_BOUND), LOWER_IMAGE_BOUND)
    return None


def deep_dream_static_image(config, img=None):
    model = intialize_model(config['model_name'], config['pretrained_weights'])

    try:
        layer_ids_to_use = [model.layer_names.index(layer_name) for layer_name in config['layers_to_use']]
    except:  # making sure you set the correct layer name for this specific model
        print(f'Invalid layer names {[layer_name for layer_name in config["layers_to_use"]]}.')
        print(f'Available layers for model {config["model_name"]} are {model.layer_names}.')

    if img is None:  # load either the provided image or start from a pure noise image
        img_path = config['input']
        # load a numpy, [0, 1] range, channel-last, RGB image
        img = load_image(img_path, target_shape=config['img_width'])
        if config['use_noise']:
            shape = img.shape
            img = np.random.uniform(low=0.0, high=1.0, size=shape).astype(np.float32)

    img_old = deepcopy(img)
    img = pre_process_numpy_img(img)
    original_shape = img.shape[:-1]  # save initial height and width

    # Note: simply rescaling the whole result (and not only details, see original implementation) gave me better results
    # Going from smaller to bigger resolution (from pyramid top to bottom)
    i_iter = 1
    for pyramid_level in range(config['pyramid_size']):
        print("Pyramid Iteration {} ... ".format(i_iter))
        new_shape = get_new_shape(config, original_shape, pyramid_level)
        img = cv.resize(img, (new_shape[1], new_shape[0]))  # resize depending on the current pyramid level
        input_tensor = pytorch_input_adapter(img)  # convert to trainable tensor
        i_iter += 1
        j_iter = 1
        for iteration in range(config['num_gradient_ascent_iterations']):
            print("\tGradient Iteration {} ... ".format(j_iter))
            # Introduce some randomness, it will give us more diverse results especially when you're making videos
            h_shift, w_shift = np.random.randint(-config['spatial_shift_size'], config['spatial_shift_size'] + 1, 2)
            input_tensor = random_circular_spatial_shift(input_tensor, h_shift, w_shift)
            # This is where the magic happens, treat it as a black box until the next cell
            gradient_ascent(config, model, input_tensor, layer_ids_to_use, iteration)
            # Roll back by the same amount as above (hence should_undo=True)
            input_tensor = random_circular_spatial_shift(input_tensor, h_shift, w_shift, should_undo=True)
            j_iter += 1
        img = pytorch_output_adapter(input_tensor)
        img = config['alpha'] * img + (1 - config['alpha']) * cv.resize(img_old, (new_shape[1], new_shape[0]))
    img = post_process_numpy_img(img)
    return img


# Run
####################################################################################################
if __name__ == "__main__":
    img = deep_dream_static_image(config)
    config['should_display'] = True
    dump_path = save_and_maybe_display_image(config, img)
    print(f'Saved DeepDream static image to: {os.path.relpath(dump_path)}\n')

