
# Train and Prune Layer by Layer:
CUDA_VISIBLE_DEVICES="0"  python train_and_prune.py --checkpoint _baseline --prune_layers [conv5_1] --layer_name conv5_1 &> output_VGG16_prune_conv5_1

CUDA_VISIBLE_DEVICES="0"  python train_and_prune.py --checkpoint _conv5_1 --prune_layers [conv5_1, conv4_3] --layer_name conv4_3 &> output_VGG16_prune_conv4_3

CUDA_VISIBLE_DEVICES="0"  python train_and_prune.py --checkpoint _conv4_3 --prune_layers [conv5_1, conv4_3, conv4_2] --layer_name conv4_2 &> output_VGG16_prune_conv4_2

CUDA_VISIBLE_DEVICES="0"  python train_and_prune.py --checkpoint _conv4_2 --prune_layers [conv5_1, conv4_3, conv4_2, conv4_1] --layer_name conv4_1 &> output_VGG16_prune_conv4_1

CUDA_VISIBLE_DEVICES="0"  python train_and_prune.py --checkpoint _conv4_1 --prune_layers [conv5_1, conv4_3, conv4_2, conv4_1, conv3_3] --layer_name conv3_3 &> output_VGG16_prune_conv3_3

CUDA_VISIBLE_DEVICES="0"  python train_and_prune.py --checkpoint _conv3_3 --prune_layers [conv5_1, conv4_3, conv4_2, conv4_1, conv3_3, conv3_2] --layer_name conv3_2 &> output_VGG16_prune_conv3_2

CUDA_VISIBLE_DEVICES="0"  python train_and_prune.py --checkpoint _conv3_2 --prune_layers [conv5_1, conv4_3, conv4_2, conv4_1, conv3_3, conv3_2, conv3_1] --layer_name conv3_1 &> output_VGG16_prune_conv3_1

CUDA_VISIBLE_DEVICES="0"  python train_and_prune.py --checkpoint _conv3_1 --prune_layers [conv5_1, conv4_3, conv4_2, conv4_1, conv3_3, conv3_2, conv3_1, conv2_2] --layer_name conv2_2 &> output_VGG16_prune_conv2_2

CUDA_VISIBLE_DEVICES="0"  python train_and_prune.py --checkpoint _conv2_2 --prune_layers [conv5_1, conv4_3, conv4_2, conv4_1, conv3_3, conv3_2, conv3_1, conv2_2, conv2_1] --layer_name conv2_1 &> output_VGG16_prune_conv2_1

CUDA_VISIBLE_DEVICES="0"  python train_and_prune.py --checkpoint _conv2_1 --prune_layers [conv5_1, conv4_3, conv4_2, conv4_1, conv3_3, conv3_2, conv3_1, conv2_2, conv2_1, conv1_2] --layer_name conv1_2 &> output_VGG16_prune_conv1_2

CUDA_VISIBLE_DEVICES="0"  python train_and_prune.py --checkpoint _conv1_2 --prune_layers [conv5_1, conv4_3, conv4_2, conv4_1, conv3_3, conv3_2, conv3_1, conv2_2, conv2_1, conv1_2, conv1_1] --layer_name conv1_1 &> output_VGG16_prune_conv1_1