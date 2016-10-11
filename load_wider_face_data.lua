require 'torch'
require 'nn'
require 'cudnn'
require 'cunn'
require 'module/normalConv'
require 'module/normalLinear'
require 'sys'
require "torch"
require "image"
require "math"
require "LuaXML"


function load_data(mode)
  target = {}
  name = {}
  if mode == "train" then
    print("training data loading...")
  end
    
  if mode == "train" then  
    file_path = "/home/gavinpan/workspace/dataset/wider_face/wider_face_train.txt"
    len       = 158988
    file_dir  = "/home/gavinpan/workspace/dataset/wider_face/WIDER_train/images/"   
  end
    
  print("file_path:" .. file_path)
  print("file_dir :" .. file_dir)
      
  fp  = io.open(file_path)

  fid = 1
  target = {}
  name = {}

  for line in fp:lines() do
    if fid % 1000 == 0 then
      print("processing gt imgs: "..fid)
    end    
    img_name, left, top, width, height = line:match("([^,]+) ([^,]+) ([^,]+) ([^,]+) ([^,]+)")    
    --[===[    
    print(img_name)
    print(left)
    print(top)
    print(width)
    print(height)
    --]===]
    left   = tonumber(left)
    top    = tonumber(top)
    width  = tonumber(width)
    height = tonumber(height)

    path = file_dir .. img_name
    img  = image.load(path)
    h    = img:size()[2]
    w    = img:size()[3]
    
    xmin = left + 1
    xmax = left + width
    ymin = top + 1 
    ymax = top + height
    if xmin < 1 then
      xmin = 1
    end
    if ymin < 1 then
      ymin = 1
    end
    if xmax > w then
      xmax = w
    end
    if ymax > h then
      ymax = h
    end
    --print("h:" .. h .. " w:" .. w)
    --print("xmin:" .. xmin .. " ymin:" .. ymin .. " xmax:" .. xmax .. " ymax:" .. ymax)
    
    -- for first save 
    if next(name) == nil then
      table.insert(name, path) 
      target_per_sample = {}
      table.insert(target_per_sample, {xmax, xmin, ymax, ymin, w, h})
      goto next_sample
    end
    
    --for k, v in ipairs(name) do
      if path == name[#name] then
        table.insert(target_per_sample, {xmax, xmin, ymax, ymin, w, h})
      else
        table.insert(name, path)    
        table.insert(target, target_per_sample)        
        target_per_sample = {}
        table.insert(target_per_sample, {xmax, xmin, ymax, ymin, w, h})
      end
    --end
    if fid == len then
      table.insert(target, target_per_sample)
    end
    ::next_sample::
    fid = fid+1
  end
  fp:close()
  return target, name
end

--4952
trainTarget_path = "./wider_face/wider_face_train_target.t7"
trainName_path = "./wider_face/wider_face_train_name.t7"
trainTarget, trainName = load_data("train")
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
