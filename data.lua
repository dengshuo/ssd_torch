require "torch"
require "image"
require "math"
require "LuaXML"


function label_to_num(label)
    if label == "aeroplane" then
        return 1
    elseif label == "bicycle" then
        return 2
    elseif label == "bird" then
        return 3
    elseif label == "boat" then
        return 4
    elseif label == "bottle" then
        return 5
    elseif label == "bus" then
        return 6
    elseif label == "car" then
        return 7
    elseif label == "cat" then
        return 8
    elseif label == "chair" then
        return 9
    elseif label == "cow" then
        return 10
    elseif label == "diningtable" then
        return 11
    elseif label == "dog" then
        return 12
    elseif label == "horse" then
        return 13
    elseif label == "motorbike" then
        return 14
    elseif label == "person" then
        return 15
    elseif label == "pottedplant" then
        return 16
    elseif label == "sheep" then
        return 17
    elseif label == "sofa" then
        return 18
    elseif label == "train" then
        return 19
    elseif label == "tvmonitor" then
        return 20
    end    
end

function load_data(mode,did)
  target = {}
  name = {}
  if mode == "train" then
    print("training data loading...")
  elseif mode == "test" then
    print("testing data loading...")
  end
    
  if mode == "train" then
    if did == 1 then 
      db_dir_ = db_dir .. "VOC0712/"  
      file_path = db_dir_ .. "ImageSets/Main/trainval.txt"     
    end
    if did == 2 then 
      db_dir_ = db_dir .. "VOC2007/" 
      file_path = db_dir_ .. "ImageSets/Main/trainval.txt"
    end
    if did == 3 then 
      db_dir_ = db_dir .. "VOC2012/" 
      file_path = db_dir_ .. "ImageSets/Main/trainval.txt"
    end
    elseif mode == "test" and did == nil then
      db_dir_ = db_dir .. "VOC2007/"
      file_path = db_dir_ .. "ImageSets/Main/test.txt"
    end
    
    print("dataset dir:"..db_dir_)
    imgDir = db_dir_ .. 'JPEGImages/'
    annotDir = db_dir_ .. 'annotations_parsed/'
    
    annotFileList = {}
    fp = io.open(file_path)
    for ann_name in fp:lines() do
      --print(annotDir .. ann_name .. ".txt")
      local fp1 = io.open(annotDir .. ann_name .. ".txt","rb")
      if fp1 then
        table.insert(annotFileList,ann_name .. ".txt")  
        fp1:close() 
      end
    end
  
    print("imgs number:"..#annotFileList)
    for fid = 1,#annotFileList do            
      if fid % 1000 == 0 then
        --print("processing gt imgs: "..fid)
        print("processing gt imgs: "..fid/#annotFileList)
      end
      --img load
      --print(annotFileList[fid]:sub(1,-4))
      --print(ann_name) 
                       
      img = image.load(imgDir .. annotFileList[fid]:sub(1,-4) .. "jpg")
      local imgHeight = img:size()[2]
      local imgWidth  = img:size()[3]
      --img = image.scale(img,imgSz,imgSz)
      --name save
      table.insert(name,imgDir .. annotFileList[fid]:sub(1,-4) .. "jpg")

      --label save
      target_per_sample = {}
      for line in io.lines(annotDir .. annotFileList[fid]) do        
        label, xmax, xmin, ymax, ymin = line:match("([^,]+),([^,]+),([^,]+),([^,]+),([^,]+)")
        --[===[
        print(label)
        print(xmax)
        print(xmin)
        print(ymax)
        print(ymin)
        --]===]
        label = label_to_num(label)
        xmax = tonumber(xmax)
        xmin = tonumber(xmin)
        ymax = tonumber(ymax)
        ymin = tonumber(ymin)
            
        --for debug
        --img = drawRectangle(img,xmin,ymin,xmax,ymax)
        --img_dir = "./gt/"
        --image.save(img_dir .. tostring(fid) .. ".jpg",img) 
        --[===[ 
        --]===]
                
        table.insert(target_per_sample,{label,xmax,xmin,ymax,ymin,imgWidth,imgHeight})
      end           
      table.insert(target,target_per_sample)
    end       
  return target, name
end
