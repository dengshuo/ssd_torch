--db_dir = "/media/sda1/Data/PASCAL_VOC/VOCdevkit/"
db_dir = "/home/gavinpan/workspace/VOCdevkit/"
--result_dir = "/media/sda1/Data/PASCAL_VOC/VOCdevkit/results/VOC2012/Main/"
result_dir = "./result/"
model_dir = result_dir .. "model/"
fig_dir = result_dir .. "fig/"

-- whether training model from scratch
continue = false
continue_iter = 0

-- class number
classNum = 21
-- channel
inputDim = 3
-- img size
imgSz = 300
-- training sample number
trainSz = 5011
-- trainSz = 17125 + 5011 + 4952
-- thr
thr = 0.7
classList = {"aeroplane","bicycle","bird","boat","bottle","bus","car","cat","chair","cow","diningtable","dog","horse","motorbike","person","pottedplant","sheep","sofa","train","tvmonitor"}
-- number of feature map
m = 5
-- scale
scale_table = {}
for k = 1,m do
  table.insert(scale_table, 0.2 + (0.95 - 0.2)/(m-1) * (k-1))
end
-- aspect ratios
ar_table = {1,2,3,1/2,1/3}
-- feaure map size
fmSz = {19,10,5,3,1}

lr = 1e-3
wDecay = 5e-4
mmt = 9e-1
--batch size
batchSz = 32
iterLimit = 6e4 - continue_iter
iterLrDecay = 4e4 - continue_iter

function str_split(inputstr, sep)
  if sep == nil then
    sep = "%s"
  end
  local t={} 
  local i=1
  for str in string.gmatch(inputstr, "([^"..sep.."]+)") do
    t[i] = str
    i = i + 1
  end
  return t
end

function table.getn(x) 
  local ret=0
  for i in pairs(x) do 
    ret=ret+1 
  end 
  return ret 
end
  
restored_box = {} --xmax xmin ymax ymin
table.insert(restored_box,torch.Tensor(6,4,fmSz[1],fmSz[1]):zero())
table.insert(restored_box,torch.Tensor(6,4,fmSz[2],fmSz[2]):zero())
table.insert(restored_box,torch.Tensor(6,4,fmSz[3],fmSz[3]):zero())
table.insert(restored_box,torch.Tensor(6,4,fmSz[4],fmSz[4]):zero())
table.insert(restored_box,torch.Tensor(5,4,fmSz[5],fmSz[5]):zero())

for lid = 1,m do       
   for r = 1,fmSz[lid] do
       for c = 1,fmSz[lid] do
            -- center of each default box
            local xCenter = (c-1+0.5)/fmSz[lid]
            local yCenter = (r-1+0.5)/fmSz[lid]

            for aid = 1,table.getn(ar_table)+1 do
                if lid < m then
                    if aid <= table.getn(ar_table) then
                        ar_factor = ar_table[aid]
                        scale_factor = scale_table[lid]
                    else
                        ar_factor = 1
                        scale_factor = math.sqrt(scale_table[lid] * scale_table[lid+1])
                    end
                else
                    if aid <= table.getn(ar_table) then
                        ar_factor = ar_table[aid]
                        scale_factor = scale_table[lid]
                    else
                        goto nextCell
                    end
                end

                local width  = scale_factor*math.sqrt(ar_factor)
                local height = scale_factor/math.sqrt(ar_factor)
                
                -- xmax
                restored_box[lid][aid][1][r][c] = math.min((xCenter + width/2)  * (imgSz),imgSz)
                -- xmin
                restored_box[lid][aid][2][r][c] = math.max((xCenter - width/2)  * (imgSz),1)
                -- ymax
                restored_box[lid][aid][3][r][c] = math.min((yCenter + height/2) * (imgSz),imgSz)
                -- ymin
                restored_box[lid][aid][4][r][c] = math.max((yCenter - height/2) * (imgSz),1)

                ::nextCell::
            end
        end
    end
end
--[===[
--]===]
function drawRectangle(img,xmin,ymin,xmax,ymax)    
  img_origin = img:clone()
  img[1][{{ymin,ymax},{xmin,xmax}}] = 255
  img[2][{{ymin,ymax},{xmin,xmax}}] = 0
  img[3][{{ymin,ymax},{xmin,xmax}}] = 0
  if ymin+2 < ymax-2 then
    ymin = ymin+2
    ymax = ymax-2
  end
  if xmin+2 < xmax-2 then
    xmin = xmin+2
    xmax = xmax-2
  end
  img[1][{{ymin,ymax},{xmin,xmax}}] = img_origin[1][{{ymin,ymax},{xmin,xmax}}]
  img[2][{{ymin,ymax},{xmin,xmax}}] = img_origin[2][{{ymin,ymax},{xmin,xmax}}]
  img[3][{{ymin,ymax},{xmin,xmax}}] = img_origin[3][{{ymin,ymax},{xmin,xmax}}]
  return img
end

