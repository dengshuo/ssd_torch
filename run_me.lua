require 'torch'
require 'cunn'

torch.setdefaulttensortype('torch.FloatTensor')
--math.randomseed(os.time())
math.randomseed(123)
dofile "data.lua"
dofile "test.lua"

--local train_model_name = "300x300_1"
local train_model_name = "300x300_2"

if train_model_name == "300x300_1" then
    dofile "etc_300_1.lua"
    dofile "model_300_1.lua"
    dofile "train_300_1.lua"
end
if train_model_name == "300x300_2" then
    dofile "etc_300_2.lua"
    dofile "model_300_2.lua"
    dofile "train_300_2.lua"
end

dateTable = os.date("*t")
resultDir =  "./result/" .. tostring(dateTable.year) .. "_" .. tostring(dateTable.month) .. "_" .. tostring(dateTable.day) .. "_" .. tostring(dateTable.hour) .. "_" .. tostring(dateTable.min) .. "_" .. tostring(dateTable.sec)
os.execute("mkdir " .. resultDir)

if mode == "train" then
    if continue == true then
        print("model loading...")
        model = torch.load(model_dir .. 'model.net')
    end

    trainTarget_path = "./voc_data/voc0712_trainval_target.t7"
    trainName_path   = "./voc_data/voc0712_trainval_name.t7"

    trainTarget  = torch.load(trainTarget_path)
    trainName    = torch.load(trainName_path)

    print("training data label:"..#trainTarget)
    print("training data names:"..#trainName)

    while tot_iter <= iterLimit do
        train(trainTarget, trainName)
    end
end

if mode == "test" then
    print("model loading...")
    model = torch.load(model_dir .. 'model.net')
    -- testTarget, testName = load_data("test")
    testTarget_path  = "./voc_data/voc2007_test_target.t7"
    testName_path    = "./voc_data/voc2007_test_name.t7"

    testTarget  = torch.load(testTarget_path)
    testName    = torch.load(trainName_path)

    print("training data label:"..#testTarget)
    print("training data names:"..#testName)

    test(testTarget,testName)
end

