

function loadbaseweight(tw)
    tw[1] = torch.load('weights/1_conv1.t7')
    tw[2] = torch.load('weights/2_conv1.t7')
    tw[3] = torch.load('weights/2_conv2.t7')
    tw[4] = torch.load('weights/2_conv3.t7')
    tw[5] = torch.load('weights/2_conv4.t7')
    tw[6] = torch.load('weights/2_conv5.t7')
    tw[7] = torch.load('weights/3_conv1.t7')
    tw[8] = torch.load('weights/3_conv2.t7')
    tw[9] = torch.load('weights/3_conv3.t7')
    tw[10] = torch.load('weights/3_conv4.t7')
    tw[11] = torch.load('weights/3_conv5.t7')
    tw[12] = torch.load('weights/4_conv1.t7')
    tw[13] = torch.load('weights/4_conv2.t7')
    tw[14] = torch.load('weights/4_conv3.t7')
    tw[15] = torch.load('weights/4_conv4.t7')
    tw[16] = torch.load('weights/4_conv5.t7')
end
function add_momentum_to_all_layer(model, tw)
    meta_momentum(model:get(1):get(1).weight, tw[1])
    k = 2
    meta_momentum(model:get(1):get(k):get(1):get(3):get(1):get(1).weight, tw[2])
    meta_momentum(model:get(1):get(k):get(1):get(3):get(1):get(4).weight, tw[3])
    meta_momentum(model:get(1):get(k):get(1):get(3):get(2).weight, tw[4])
    meta_momentum(model:get(1):get(k):get(2):get(1):get(1):get(3).weight, tw[5])
    meta_momentum(model:get(1):get(k):get(2):get(1):get(1):get(6).weight, tw[6])
    k = 3
    meta_momentum(model:get(1):get(k):get(1):get(3):get(1):get(1).weight, tw[7])
    meta_momentum(model:get(1):get(k):get(1):get(3):get(1):get(4).weight, tw[8])
    meta_momentum(model:get(1):get(k):get(1):get(3):get(2).weight, tw[9])
    meta_momentum(model:get(1):get(k):get(2):get(1):get(1):get(3).weight, tw[10])
    meta_momentum(model:get(1):get(k):get(2):get(1):get(1):get(6).weight, tw[11])
    k = 4
    meta_momentum(model:get(1):get(k):get(1):get(3):get(1):get(1).weight, tw[12])
    meta_momentum(model:get(1):get(k):get(1):get(3):get(1):get(4).weight, tw[13])
    meta_momentum(model:get(1):get(k):get(1):get(3):get(2).weight, tw[14])
    meta_momentum(model:get(1):get(k):get(2):get(1):get(1):get(3).weight, tw[15])
    meta_momentum(model:get(1):get(k):get(2):get(1):get(1):get(6).weight, tw[16])
end


-- override print to always flush the output
local old_print = print
local print = function(...)
    old_print(...)
    io.flush()
end
