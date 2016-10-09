require 'torch'
require 'nn'
require 'cudnn'
require 'cunn'
require 'module/normalConv'
require 'module/normalLinear'
require 'sys'
require "etc_300_1.lua"
dofile 'data.lua'

--16551
--[===[
trainTarget_path = "./voc_data/voc0712_trainval_target.t7"
trainName_path = "./voc_data/voc0712_trainval_name.t7"
trainTarget, trainName = load_data("train",1)
--]===]

--5011
--[===[
trainTarget_path = "./voc_data/voc2007_trainval_target.t7"
trainName_path = "./voc_data/voc2007_trainval_name.t7"
trainTarget, trainName = load_data("train",2)
--]===]

--11540
--[===[
trainTarget_path = "./voc_data/voc2012_trainval_target.t7"
trainName_path = "./voc_data/voc2012_trainval_name.t7"
trainTarget, trainName = load_data("train",3)
--]===]

--4952
trainTarget_path = "./voc_data/voc2007_test_target.t7"
trainName_path = "./voc_data/voc2007_test_name.t7"
trainTarget, trainName = load_data("test")
--[===[
--]===]

print("training data label:"..#trainTarget)
print("training data names:"..#trainName)
torch.save(trainTarget_path, trainTarget)
torch.save(trainName_path, trainName)

trainTarget = torch.load(trainTarget_path)
trainName = torch.load(trainName_path)
print("training data label:"..#trainTarget)
print("training data names:"..#trainName)
