require 'torch'
require 'image'
--file_path = "mscnn_wider_face_train.txt"
--id = 12879 + 1

file_path = "mscnn_wider_face_val.txt"
id = 3225 + 1
math.randomseed(os.time())
--math.randomseed(123)
fp  = io.open(file_path, 'a')
bg_path = "/home/gavinpan/workspace/dataset/bg/"
fp1 = io.popen("ls " .. bg_path)
for line in fp1:lines() do
  fp:write("# " .. tostring(id) .. "\n")
  img = image.load(bg_path .. line)
  fp:write(bg_path .. line .. "\n")
  c = img:size()[1]
  h = img:size()[2]
  w = img:size()[3]
  fp:write(tostring(c) .. "\n")
  fp:write(tostring(h) .. "\n")
  fp:write(tostring(w) .. "\n")
  print(c .. " " .. h .. " " .. w)
  fp:write(tostring(0) .. "\n") 
  num = math.random(20)
  print(num)  
  fp:write(tostring(num) .. "\n") 
  for i = 1, num do
    x1 = math.floor(math.random(0, w-1))
    y1 = math.floor(math.random(0, h-1))
    w1 = math.floor(math.random(w*0.1, w-1))
    h1 = math.floor(math.random(w*0.1, h-1))
    -- print(x1 .. " " .. y1 .. " " .. x2 .. " " .. y2)
    while x1 + w1 >= w-1 do
      x1 = math.floor(math.random(0, w-1))
      w1 = math.floor(math.random(w*0.1, w-1))
    end
    while y1 + h1 >= h-1 do
      y1 = math.floor(math.random(0, h-1))
      h1 = math.floor(math.random(w*0.1, h-1))
    end
    print(x1 .. " " .. y1 .. " " .. x1 + w1 .. " " .. y1 + h1) 
    fp:write(tostring(x1) .. " " .. tostring(y1) .. " " .. tostring(x1 + w1) .. " " .. tostring(y1 + h1) .. "\n")
  end
  id = id + 1
end

