-- how to run: th train.lua

-- Setting required packages
require 'nn';

-- Download data from files, and initialize it
trainset = torch.load('cifar10-train.t7')
testset = torch.load('cifar10-test.t7')
classes = {'airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck'}
print('Initialize completed')

--print(trainset)
--[[
{
    data : ByteTensor - size: 10000x3x32x32
    label : ByteTensor - size: 10000
}
]]

--print(#trainset.data)
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

--print(trainset:size())
-- 10000

-- Normalization of trainset.data
mean = {}
stdv  = {}
for i=1,3 do
    mean[i] = trainset.data[{ {}, {i}, {}, {}  }]:mean()
    print('Channel ' .. i .. ', Mean: ' .. mean[i])
    trainset.data[{ {}, {i}, {}, {}  }]:add(-mean[i])
    
    stdv[i] = trainset.data[{ {}, {i}, {}, {}  }]:std()
    print('Channel ' .. i .. ', Standard Deviation: ' .. stdv[i])
    trainset.data[{ {}, {i}, {}, {}  }]:div(stdv[i])
end
print('data normalization completed')

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

print('Lenet5\n' .. net:__tostring());