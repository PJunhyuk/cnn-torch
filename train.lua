-- how to run: th train.lua

trainset = torch.load('cifar10-train.t7')
testset = torch.load('cifar10-test.t7')
classes = {'airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck'}

print(trainset)
--[[
{
    data : ByteTensor - size: 10000x3x32x32
    label : ByteTensor - size: 10000
}
]]

print(#trainset.data)
--[[
10000
3
32
32
[torch.LongStorage of size 4]
]]

setmetatable(trainset,
    {__index = function(t, i)
        return {t.data[i], t.label[i]}
    end}
);

trainset.data = trainset.data:double()

function trainset:size()
    return self.data:size(1)
end

print(trainset:size())