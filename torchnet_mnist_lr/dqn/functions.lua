

function loadbaseweight(tw)
    tw[1] = torch.load('weights/w2.t7')
    tw[2] = torch.load('weights/w5.t7')
    tw[3] = torch.load('weights/w9.t7')
end
function add_momentum_to_all_layer(model, tw)
    meta_momentum(model:get(2).weight, tw[1])
    meta_momentum(model:get(5).weight, tw[2])
    meta_momentum(model:get(9).weight, tw[3])
end

meta_momentum_coefficient = 0.01

function meta_momentum(w, targetw)
    local tmp = torch.CudaTensor(targetw:size()):copy(targetw)
    w:add(tmp:add(-w):mul(meta_momentum_coefficient))  --w = w + (target_w - w) * co-efficient
end


-- override print to always flush the output
local old_print = print
local print = function(...)
    old_print(...)
    io.flush()
end
