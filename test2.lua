require 'image'
img = image.load('0_Parade_marchingband_1_5.jpg')
h  = img:size()[2]
w  = img:size()[3]
print("h:" .. h .. " w:" .. w)

target = {1}
print(target)
print(next(target) == nil)

for k, v in ipairs(target) do
  print(k .. " " .. v)
end
