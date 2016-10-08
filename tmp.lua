require 'torch'
require 'nn'
require 'cudnn'
require 'cunn'
require 'module/normalConv'
require 'module/normalLinear'
require 'sys'
dofile 'etc.lua'
dofile 'data.lua'

--a = {}
--print(type(a))
--for i = 3,3 do
--  print(i)
--end

trainTarget_path = "voc0712_trainTarget.t7"
trainName_path = "voc0712_trainName.t7"
--[===[
--]===]
trainTarget, trainName = load_data("train",1)
print("training data label:"..#trainTarget)
print("training data names:"..#trainName)
torch.save(trainTarget_path, trainTarget)
torch.save(trainName_path, trainName)

trainTarget = torch.load(trainTarget_path)
trainName = torch.load(trainName_path)
print("training data label:"..#trainTarget)
print("training data names:"..#trainName)


goto next1
::next1::
function table.getn(x) 
  local ret=0
  for i in pairs(x) do 
    ret=ret+1 
  end 
  return ret 
end
ar_table = {1,2,3,1/2,1/3}
print(#ar_table)
print(table.getn(ar_table))
print(ar_table)
