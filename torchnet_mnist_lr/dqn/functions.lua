

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
momentum_times = 33 --390*3  -- 3 epoch

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

function kurtosis(t)
    local m = t:mean()
    local n = t:size()[1]
    local cu_t = t:cuda()
    local sqSum = cu_t:csub(m):pow(2):sum()
    local fourthSum = cu_t:csub(m):pow(4):sum()
    --[[for i = 1, n do
        local sq = (t[i] - m)*(t[i] - m)
        sqSum = sqSum + sq
        fourthSum = fourthSum + sq*sq
    end]]
    return n*fourthSum/(sqSum*sqSum) - 3
end

function skewness(t)
    local m = t:mean()
    local n = t:size()[1]
    local cu_t = t:cuda()
    local sqSum = cu_t:csub(m):pow(2):sum()
    local thirdSum = cu_t:csub(m):pow(3):sum()
    --[[for i = 1, n do
        sqSum = sqSum + (t[i] - m)*(t[i] - m)
        thirdSum = thirdSum + (t[i] - m)*(t[i] - m)*(t[i] - m)
    end]]
    return (thirdSum/n)/math.pow(sqSum/(n - 1), 3/2)
    --return thirdSum_tensor:div(n):div(sqSum_tensor:div(n-1):pow(3/2))
end

function central_moment(t, k)
    local t_k = torch.pow(t, k)
    return t_k:mean()
end

function k_bins_entropy(t)
    local k = 32
    local max_ = torch.max(t)
    local min_ = torch.min(t)
    local vec = torch.zeros(k)
    return vec
    --[[local step = (max_ - min_) / k
    if step == 0 then
        vec[32] = t:nElement()
        return vec
    end
    local index = torch.floor(t:csub(min_):div(step)):add(1)
    print('min_ = ' .. min_)
    print('step = ' .. step)
    --print(index)
    for i = 1, index:nElement() do
        local idx = index[i]
        if idx > 32 then idx = 32 end
        assert(idx > 0 and idx <= 32)
        vec[idx] = vec[idx] + 1
    end
    return vec]]
end
