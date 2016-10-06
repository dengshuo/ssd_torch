require 'torch'
require 'cunn'

torch.setdefaulttensortype('torch.FloatTensor')
--math.randomseed(os.time())
math.randomseed(123)

dofile "etc.lua"
dofile "data.lua"
dofile "model.lua"
dofile "train.lua"
dofile "test.lua"

dateTable = os.date("*t")
resultDir =  "./result/" .. tostring(dateTable.year) .. "_" .. tostring(dateTable.month) .. "_" .. tostring(dateTable.day) .. "_" .. tostring(dateTable.hour) .. "_" .. tostring(dateTable.min) .. "_" .. tostring(dateTable.sec)
os.execute("mkdir " .. resultDir)

if mode == "train" then
    if continue == true then
        print("model loading...")
        model = torch.load(model_dir .. 'model.net')
    end
    trainTarget_path = "./voc_data/voc2007_trainTarget.t7"
    trainName_path   = "./voc_data/voc2007_trainName.t7"  

    trainTarget = torch.load(trainTarget_path)
    trainName   = torch.load(trainName_path)
    
    print("training data label:"..#trainTarget)
    print("training data names:"..#trainName)
    
    while tot_iter <= iterLimit do
        train(trainTarget, trainName)
    end
end

if mode == "test" then    
    print("model loading...")
    model = torch.load(model_dir .. 'model.net')
    testTarget, testName = load_data("test")
    test(testTarget,testName)
end

