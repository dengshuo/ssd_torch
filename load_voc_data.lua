require 'torch'
require 'nn'
require 'cudnn'
require 'cunn'
require 'module/normalConv'
require 'module/normalLinear'
require 'sys'
dofile 'etc.lua'
dofile 'data.lua'
--27088
--[===[
trainTarget_path = "./voc_data/voc0712_trainTarget.t7"
trainName_path = "./voc_data/voc0712_trainName.t7"
trainTarget, trainName = load_data("train",1)
--]===]
--9963
--[===[
trainTarget_path = "./voc_data/voc2007_trainTarget.t7"
trainName_path = "./voc_data/voc2007_trainName.t7"
trainTarget, trainName = load_data("train",2)
--]===]
--17125
trainTarget_path = "./voc_data/voc2012_trainTarget.t7"
trainName_path = "./voc_data/voc2012_trainName.t7"
trainTarget, trainName = load_data("train",3)

print("training data label:"..#trainTarget)
print("training data names:"..#trainName)
torch.save(trainTarget_path, trainTarget)
torch.save(trainName_path, trainName)

trainTarget = torch.load(trainTarget_path)
trainName = torch.load(trainName_path)
print("training data label:"..#trainTarget)
print("training data names:"..#trainName)
