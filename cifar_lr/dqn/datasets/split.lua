data = torch.load('cifar10_whitened.t7')

v = {}
for i = 1,5000 do
	v[i] = i
end
vl = data.trainData.labels:index(1, torch.LongTensor(v))
vd = data.trainData.data:index(1, torch.LongTensor(v))
vs = data.trainData.size

v = {}
for i = 5001,50000 do
	v[i-5000] = i
end
tl = data.trainData.labels:index(1, torch.LongTensor(v))
td = data.trainData.data:index(1, torch.LongTensor(v))


data.trainData.data = td
data.trainData.labels = tl
data.trainData.size = function() return 45000; end

data['validationData'] = {}
data['validationData']['data'] = vd
data['validationData']['labels'] = vl
data['validationData']['size'] = function () return 5000; end

print(data)
print(data.trainData.size())
print(data.validationData.size())

torch.save('trainvalidata.t7', data)

