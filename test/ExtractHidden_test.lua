require 'torch'
require 'nn'
require 'image'

require 'densecap.DenseCapModel'
local utils = require 'densecap.utils'
local box_utils = require 'densecap.box_utils'
local vis_utils = require 'densecap.vis_utils'


-- Load the model, and cast to the right type
local opt = {}
opt.gpu = 0
opt.use_cudnn = 1
opt.checkpoint = 'data/models/densecap/densecap-pretrained-vgg16.t7' 
opt.rpn_nms_thresh = 0.7
opt.final_nms_thresh = 0.3
opt.num_proposals = 1000
opt.image_size = 720
local dtype, use_cudnn = utils.setup_gpus(opt.gpu, opt.use_cudnn)
local checkpoint = torch.load(opt.checkpoint)
local model = checkpoint.model
model:convert(dtype, use_cudnn)
model:setTestArgs{
  rpn_nms_thresh = opt.rpn_nms_thresh,
  final_nms_thresh = opt.final_nms_thresh,
  num_proposals = opt.num_proposals,
}
model:evaluate()

-- evaluate
local img_path = 'imgs/elephant.jpg' 
local img = image.load(img_path, 3)
img = image.scale(img, opt.image_size):float()
local H, W = img:size(2), img:size(3)
local img_caffe = img:view(1, 3, H, W)
img_caffe = img_caffe:index(2, torch.LongTensor{3, 2, 1}):mul(255)
local vgg_mean = torch.FloatTensor{103.939, 116.779, 123.68}
vgg_mean = vgg_mean:view(1, 3, 1, 1):expand(1, 3, H, W)
img_caffe:add(-1, vgg_mean)
img_caffe = img_caffe:type(dtype)

-- get all outputs from input
local output = model:extractAllFeatures(img_caffe)
local seqs = output[3]
local roi_codes = output[4]
local hidden_codes = output[5]
local captions = output[6]

-- -- get hidden outputs from roi_codes
-- local hidden_codes2 = model.nets.language_model:extract_hidden(roi_codes, seqs)

-- -- print(hidden_codes[{ {2}, {1, 50} }]:view(5, 10))
-- -- print(hidden_codes2[{ {2}, {1, 50} }]:view(5, 10))
-- print('Is sum of hidden_codes equal to hidden_codes2?', hidden_codes:sum() == hidden_codes2:sum())

local output2 = model:forward_test_beams(img_caffe, 5, true)
local beam_hidden_codes = output2[3]
local captions2 = output2[2]
print(captions2)

-- print(hidden_codes[{ {1}, {1, 50} }]:view(5, 10))
-- print(beam_hidden_codes[1][{ {}, {1, 50} }]:view(5, 10))
-- print(captions, captions2)













