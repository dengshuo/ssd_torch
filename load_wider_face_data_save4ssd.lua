require 'torch'
require 'nn'
require 'cudnn'
require 'cunn'
require 'sys'
require "torch"
require "image"
require "math"
require "LuaXML"

function load_data_save4ssd(mode)
  target = {}
  name = {}
  if mode == "train" then
    print("training data loading...")
  end
    
  if mode == "train" then  
    file_path = "/home/gavinpan/workspace/dataset/wider_face/wider_face_train_0.txt"
    len       = 158988 - 19
    file_dir  = "/home/gavinpan/workspace/dataset/wider_face/WIDER_train/images/"   
  end
  
  if mode == "val" then  
    file_path = "/home/gavinpan/workspace/dataset/wider_face/wider_face_val_0.txt"
    len       = 39454 - 9
    file_dir  = "/home/gavinpan/workspace/dataset/wider_face/WIDER_val/images/"   
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
    
    xmin = left 
    xmax = left + width
    ymin = top  
    ymax = top + height
    if xmin < 0 then
      xmin = 0
    end
    if ymin < 0 then
      ymin = 0
    end
    if xmax >= w then
      xmax = w-1
    end
    if ymax >= h then
      ymax = h-1
    end
    --print("h:" .. h .. " w:" .. w)
    --print("xmin:" .. xmin .. " ymin:" .. ymin .. " xmax:" .. xmax .. " ymax:" .. ymax)
    
    -- for first save 
    if next(name) == nil then
      --print(string.sub(img_name,1,-5))
      table.insert(name, string.sub(img_name,1,-5)) 
      target_per_sample = {}
      table.insert(target_per_sample, {xmin, ymin, xmax, ymax, w, h})
      goto next_sample
    end
    
    --for k, v in ipairs(name) do
      if string.sub(img_name,1,-5) == name[#name] then
        table.insert(target_per_sample, {xmin, ymin, xmax, ymax, w, h})
      else
        --print(string.sub(img_name,1,-5))
        table.insert(name, string.sub(img_name,1,-5))    
        table.insert(target, target_per_sample)        
        target_per_sample = {}
        table.insert(target_per_sample, {xmin, ymin, xmax, ymax, w, h})
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
-- 12880
-- 12861
-- train 

Target_path = "./wider_face/wider_face_train_target_ssd.t7"
Name_path   = "./wider_face/wider_face_train_name_ssd.t7"
save_dir    = "./wider_face_train/"
--[===[
target, name = load_data_save4ssd("train")

--]===]

-- val 3226
-- val 3217
--[===[
Target_path = "./wider_face/wider_face_val_target_ssd.t7"
Name_path   = "./wider_face/wider_face_val_name_ssd.t7"
save_dir    = "./wider_face_val/"

target, name = load_data_save4ssd("val")

--]===]
--[===[
print("data label:"..#target)
print("data names:"..#name)
torch.save(Target_path, target)
torch.save(Name_path, name)

--]===]
target = torch.load(Target_path)
name   = torch.load(Name_path)
print("data label:"..#target)
print("data names:"..#name)

--print(target[1])
--print(name[1])

for i = 1, #target do
  --print(name[i] .. ".txt")   
  path = string.gsub(name[i] .. ".txt", "/", "_")
  --print(path)
  fp = io.open(save_dir .. path, "w")
  --print("number of sample:" .. #target[i])
  for j = 1, #target[i] do
    local x1 = target[i][j][1]
    local y1 = target[i][j][2]
    local x2 = target[i][j][3]
    local y2 = target[i][j][4]
    local w  = target[i][j][5]
    local h  = target[i][j][6]  
    --print(x1 .. " " .. y1 .. " " .. x2 .. " " .. y2)           
    fp:write(tostring(x1) .. " " .. tostring(y1) .. " " .. tostring(x2) .. " " .. tostring(y2) .. " " .. tostring(w) .." " .. tostring(h) .. "\n")
  end
  fp:close()
end
--[===[
--]===]

