
minibatch = torch.class('MiniBatch')


function minibatch:__init()
end

function minibatch:reset(batchSize)
    self.rnum = 10
    local r = {}
    for i=1,self.rnum do
        r[i] = i
    end
    self.rouletee = torch.Tensor(r)
    self.c = {}
    for i=1,10 do
        self.c[i] = torch.load('index/CIFAR'..i) --TODO
    end
    self.mapping = {}
    for i=1,10 do self.mapping[i] = 0 end
    self.randindex = {}
    self.pointer = {}
    for i=1,10 do
        self.randindex[i] = torch.randperm(self.c[i]:nElement())
        self.pointer[i] = 1
    end
    self.batchsize = batchSize

end

function minibatch:step(action)

    for i=self.rnum,2,-1 do
        self.rouletee[i] = self.rouletee[i-1]
    end
    self.rouletee[1] = action
    --select mini-batch according to actions
    --actions number are classes number
    for i=1,self.rnum do
        if (self.rouletee[i]) then
            self.mapping[self.rouletee[i]] = self.mapping[self.rouletee[i]] + 1
        end
    end
    local res = {}
    print('actions are: ')
    for i=1,self.rnum do
        print(self.rouletee[i])
    end
    io.flush()
    local size2 = math.floor(self.batchsize / self.rnum)
    local size1 = self.batchsize - size2 * (self.rnum - 1)
    print('size2 = '.. size2)
    print('size1 = '.. size1)
    for i=1, self.rnum do
        if (self.rouletee[i]) then
            local c = self.rouletee[i]
            local size_ = size2
            if i == 1 then size_ = size1 end
            for j=1, size_ do
                if self.pointer[c] <= self.c[c]:nElement() then
                    res[#res+1] = self.c[c][self.randindex[c][self.pointer[c]]]
                    self.pointer[c] = self.pointer[c] + 1  --TODO: should reset to 1 when restart game
                end
            end
        end
    end
    if #res < self.batchsize then
        for i=1,10 do
            while self.pointer[i] <= self.c[i]:nElement() and #res < self.batchsize do
                res[#res+1] = self.c[i][self.randindex[i][self.pointer[i]]]
                self.pointer[i] = self.pointer[i] + 1  --TODO should reset to 0 when restart game
            end
            if #res == self.batchsize then break end
        end
    end
    self.batch = torch.LongTensor(res) --batch_indices to learn
    io.flush()
    return self.batch

end


function minibatch:getOneHot()
    local onehot = torch.zeros(self.rnum):float()
    for i = 1, self.rnum do
        onehot[self.rouletee[i]] = 1
    end
    return onehot
end