require 'loadcaffe'
require 'torch'
model = loadcaffe.load('VGG_ILSVRC_16_layers_deploy.prototxt', 'VGG_ILSVRC_16_layers.caffemodel')
print(model)
model_path = "./models/VGG/vgg_pretrain.t7"
torch.save(model_path, model)
