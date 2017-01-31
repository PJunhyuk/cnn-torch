-- how to run: th train.lua

-- Check time
local time = os.clock()

-- Setting required packages
require 'nn';

-- Download data from files, and initialize it
trainset = torch.load('cifar10-train.t7')
testset = torch.load('cifar10-test.t7')
classes = {'airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck'}
print('Initialize completed')

-- print(trainset)
--[[
{
    data : ByteTensor - size: 10000x3x32x32
    label : ByteTensor - size: 10000
}
]]

-- print(#trainset.data)
--[[
10000
3
32
32
[torch.LongStorage of size 4]
]]

-- Set index operator
setmetatable(trainset,
    {__index = function(t, i)
        return {t.data[i], t.label[i]}
    end}
);

-- Change data type from ByteTensor To DoubleTensor
trainset.data = trainset.data:double()

-- Set :size() of trainset
function trainset:size()
    return self.data:size(1)
end

-- print(trainset:size())
-- 10000

-- Normalize trainset.data
mean = {}
stdv  = {}
for i=1,3 do
    mean[i] = trainset.data[{ {}, {i}, {}, {}  }]:mean()
    -- print('Channel ' .. i .. ', Mean: ' .. mean[i])
    trainset.data[{ {}, {i}, {}, {}  }]:add(-mean[i])
    
    stdv[i] = trainset.data[{ {}, {i}, {}, {}  }]:std()
    -- print('Channel ' .. i .. ', Standard Deviation: ' .. stdv[i])
    trainset.data[{ {}, {i}, {}, {}  }]:div(stdv[i])
end
print('Data normalization completed')

--print(trainset.data[100])

-- Set Neural Networks
net = nn.Sequential()
net:add(nn.SpatialConvolution(3, 6, 5, 5))
net:add(nn.SpatialMaxPooling(2,2,2,2))
net:add(nn.SpatialConvolution(6, 16, 5, 5))
net:add(nn.SpatialMaxPooling(2,2,2,2))
net:add(nn.View(16*5*5))
net:add(nn.Linear(16*5*5, 120))
net:add(nn.Linear(120, 84))
net:add(nn.Linear(84, 10))
net:add(nn.LogSoftMax())
print('Neural Networks setting completed')

-- print(net:__tostring());
--[[
nn.Sequential {
  [input -> (1) -> (2) -> (3) -> (4) -> (5) -> (6) -> (7) -> (8) -> (9) -> output]
  (1): nn.SpatialConvolution(3 -> 6, 5x5)
  (2): nn.SpatialMaxPooling(2x2, 2,2)
  (3): nn.SpatialConvolution(6 -> 16, 5x5)
  (4): nn.SpatialMaxPooling(2x2, 2,2)
  (5): nn.View(400)
  (6): nn.Linear(400 -> 120)
  (7): nn.Linear(120 -> 84)
  (8): nn.Linear(84 -> 10)
  (9): nn.LogSoftMax
}
]]

-- Define cost function
criterion = nn.ClassNLLCriterion()

-- Train
trainer = nn.StochasticGradient(net, criterion)
trainer.learningRate = 0.001
trainer.maxIteration = 5 -- set epoch

trainer:train(trainset)
print('Train completed')

-- Check time
print(string.format("elapsed time until train completed: %.2f\n", os.clock() - time))

-- Normalize testset.data
testset.data = testset.data:double()
for i=1,3 do
    testset.data[{ {}, {i}, {}, {}  }]:add(-mean[i])  
    testset.data[{ {}, {i}, {}, {}  }]:div(stdv[i])
end
print('Testset data normalization completed')

-- Checking results
correct = 0
for i=1,10000 do
    local groundtruth = testset.label[i]
    local prediction = net:forward(testset.data[i])
    local confidences, indices = torch.sort(prediction, true)
    if groundtruth == indices[1] then
        correct = correct + 1
    end
end

print(correct, 100*correct/10000 .. ' % ')

-- Check time
print(string.format("total elapsed time: %.2f\n", os.clock() - time))