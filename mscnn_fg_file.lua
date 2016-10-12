require 'torch'
require 'image'

trainTarget_path = "./wider_face/wider_face_train_target.t7"
trainName_path = "./wider_face/wider_face_train_name.t7"
file_path = "mscnn_wider_face_train.txt"

--trainTarget_path = "./wider_face/wider_face_val_target.t7"
--trainName_path = "./wider_face/wider_face_val_name.t7"
--file_path = "mscnn_wider_face_val.txt"

trainTarget = torch.load(trainTarget_path)
trainName   = torch.load(trainName_path)
print("training data label:"..#trainTarget)
print("training data names:"..#trainName)
math.randomseed(123)
fp  = io.open(file_path, 'w')
for i = 1, #trainTarget do 
  fp:write("# " .. tostring(i-1) .. "\n")
  print(trainName[i])
  fp:write(trainName[i] .. "\n")
  img = image.load(trainName[i])
  c = img:size()[1]
  h = img:size()[2]
  w = img:size()[3]
  fp:write(tostring(c) .. "\n")
  fp:write(tostring(h) .. "\n")
  fp:write(tostring(w) .. "\n")
  fp:write(tostring(#trainTarget[i]) .. "\n")
  for j = 1, #trainTarget[i] do
    fp:write(tostring(1) .. " " .. tostring(0) .. " " .. tostring(trainTarget[i][j][3]-1) .. " " .. tostring(trainTarget[i][j][2]-1) .. " " .. tostring(trainTarget[i][j][5]-1) .. " " .. tostring(trainTarget[i][j][4]-1) .. "\n")
  end
  fp:write(tostring(0) .. "\n") 
end
