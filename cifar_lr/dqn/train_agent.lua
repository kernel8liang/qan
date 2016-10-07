--[[
Copyright (c) 2014 Google Inc.

See LICENSE file for full terms of limited license.
]]

if not dqn then require "initenv" end
require 'functions'
require 'config'
--CNN setting
require 'xlua'
require 'optim'
require 'image'
local tnt = require 'torchnet'
local c = require 'trepl.colorize'
local json = require 'cjson'
local utils = paths.dofile'models/utils.lua'
-- for memory optimizations and graph generation
local optnet = require 'optnet'
local graphgen = require 'optnet.graphgen'
local iterm = require 'iterm'
require 'iterm.dot'
local total_reward
local nrewards
local nepisodes
local episode_reward

local opt = opt
--- General setup.
--local game_env, game_actions, agent, opt = setup(opt)
local game_actions, agent, opt = setup(opt)

local learn_start = agent.learn_start
local start_time = sys.clock()
local reward_counts = {}
local episode_counts = {}
local time_history = {}
local v_history = {}
local qmax_history = {}
local td_history = {}
local reward_history = {}
local step = 0
time_history[1] = 0

local cnnopt = {
	dataset = './datasets/cifar10_whitened.t7',
	save = 'logs',
	batchSize = 128,
	learningRate = 0.1,
	learningRateDecay = 0,
	learningRateDecayRatio = 0.2,
	weightDecay = 0.0005,
	dampening = 0,
	momentum = 0.9,
	epoch_step = "80",
	max_epoch = 10,
	model = 'wide-resnet',
	optimMethod = 'sgd',
	init_value = 10,
	depth = 40,
	shortcutType = 'A',
	nesterov = false,
	dropout = 0,
	hflip = true,
	randomcrop = 4,
	imageSize = 32,
	randomcrop_type = 'zero',
	cudnn_deterministic = false,
	optnet_optimize = true,
	generate_graph = false,
	multiply_input_factor = 1,
	widen_factor = 1,
	nGPU = 1,
	data_type = 'torch.CudaTensor'
}
while episode < max_episode do
	--collectgarbage()
	episode = episode + 1
	local last_validation_loss = 10000
	local early_stop = false
	local meta_momentum_coefficient = 0.01
	local min_epoch = 10
	local add_momentum = 0
	local last_loss = nil
	local step_num = 0
	local log_sum = 0
	cnnopt = xlua.envparams(cnnopt)

	cnnopt.epoch_step = tonumber(cnnopt.epoch_step) or loadstring('return '..cnnopt.epoch_step)()
	print(cnnopt)

	print(c.blue '==>' ..' loading data')
	local provider = torch.load(cnnopt.dataset)
	cnnopt.num_classes = provider.testData.labels:max()

	local function cast(x) return x:type(cnnopt.data_type) end

	print(c.blue '==>' ..' configuring model')
	local model = nn.Sequential()
	local net = dofile('models/'..cnnopt.model..'.lua')(cnnopt)
	if cnnopt.data_type:match'torch.Cuda.*Tensor' then
	   require 'cudnn'
	   cudnn.fastest = true
	   require 'cunn'
	   cudnn.convert(net, cudnn):cuda()
	   cudnn.benchmark = true
	   if cnnopt.cudnn_deterministic then
		  net:apply(function(m) if m.setMode then m:setMode(1,1,1) end end)
	   end

	   print(net)
	   print('Network has', #net:findModules'cudnn.SpatialConvolution', 'convolutions')

	   local sample_input = torch.randn(8,3,cnnopt.imageSize,cnnopt.imageSize):cuda()
	   if cnnopt.generate_graph then
		  iterm.dot(graphgen(net, sample_input), cnnopt.save..'/graph.pdf')
	   end
	   if cnnopt.optnet_optimize then
		  optnet.optimizeMemory(net, sample_input, {inplace = false, mode = 'training'})
	   end

	end
	model:add(utils.makeDataParallelTable(net, cnnopt.nGPU))
	cast(model)

	local function hflip(x)
	   return torch.random(0,1) == 1 and x or image.hflip(x)
	end

	local function randomcrop(x)
	   local pad = cnnopt.randomcrop
	   if cnnopt.randomcrop_type == 'reflection' then
		  module = nn.SpatialReflectionPadding(pad,pad,pad,pad):float()
	   elseif cnnopt.randomcrop_type == 'zero' then
		  module = nn.SpatialZeroPadding(pad,pad,pad,pad):float()
	   else
		  error'unknown mode'
	   end

	   local imsize = cnnopt.imageSize
	   local padded = module:forward(x)
	   local x = torch.random(1,pad*2 + 1)
	   local y = torch.random(1,pad*2 + 1)
	   return padded:narrow(3,x,imsize):narrow(2,y,imsize)
	end


	local function getIterator(mode)
	   return tnt.ParallelDatasetIterator{
		  nthread = 1,
		  init = function()
			 require 'torchnet'
			 require 'image'
			 require 'nn'
		  end,
		  closure = function()
			 local dataset = provider[mode..'Data']

			 local list_dataset = tnt.ListDataset{
				list = torch.range(1, dataset.labels:numel()):long(),
				load = function(idx)
				   return {
					  input = dataset.data[idx]:float(),
					  target = torch.LongTensor{dataset.labels[idx]},
				   }
				end,
			 }
			 if mode == 'train' then
				return list_dataset
				   :shuffle()
				   :transform{
					  input = tnt.transform.compose{
						  cnnopt.hflip and hflip,
						  cnnopt.randomcrop > 0 and randomcrop,
					  }
				   }
				   :batch(cnnopt.batchSize, 'skip-last')
			 elseif mode == 'test' then
				return list_dataset
				   :batch(cnnopt.batchSize, 'include-last')
			 elseif mode == 'validation' then
				return list_dataset
				   :batch(cnnopt.batchSize, 'include-last')
			 end
		  end
	   }
	end

	local function log(t) print('json_stats: '..json.encode(tablex.merge(t,cnnopt,true))) end

	print('Will save at '..cnnopt.save)
	paths.mkdir(cnnopt.save)

	local engine = tnt.OptimEngine()
	local criterion = cast(nn.CrossEntropyCriterion())
	local meter = tnt.AverageValueMeter()
	local clerr = tnt.ClassErrorMeter{topk = {1}}
	local train_timer = torch.Timer()
	local test_timer = torch.Timer()

	engine.hooks.onStartEpoch = function(state)
	   local epoch = state.epoch + 1
	   print('==>'.." online epoch # " .. epoch .. ' [batchSize = ' .. cnnopt.batchSize .. ']')
	   meter:reset()
	   clerr:reset()
	   train_timer:reset()
	   if torch.type(cnnopt.epoch_step) == 'number' and epoch % cnnopt.epoch_step == 0 or
			torch.type(cnnopt.epoch_step) == 'table' and tablex.find(cnnopt.epoch_step, epoch) then
			cnnopt.learningRate = cnnopt.learningRate * cnnopt.learningRateDecayRatio
			state.config = tablex.deepcopy(cnnopt)
			state.optim = tablex.deepcopy(cnnopt)
	   end
	end

	engine.hooks.onEndEpoch = function(state)
		local train_loss = meter:value()
		local train_err = clerr:value{k = 1 }
		os.execute('echo ' .. (100 - train_err) .. '>> ' .. train_output_file)
		local train_time = train_timer:time().real
		meter:reset()
		clerr:reset()
		test_timer:reset()

		curr_mode = 'testcnn'
		engine:test{
			network = model,
			iterator = getIterator('test'),
			criterion = criterion,
		}
		curr_mode = 'traincnn'

		log{
			loss = train_loss,
			train_loss = train_loss,
			train_acc = 100 - train_err,
			epoch = state.epoch,
			test_acc = 100 - clerr:value{k = 1},
			lr = cnnopt.learningRate,
			train_time = train_time,
			test_time = test_timer:time().real,
			n_parameters = state.params:numel(),
		}
		os.execute('echo ' .. (100 - clerr:value{k = 1}) .. '>> ' .. output_file)

		if state.epoch == cnnopt.max_epoch then
			os.execute('echo ------- >> ' .. train_output_file)
			os.execute('echo ------- >> ' .. output_file)
			os.execute('echo ------- >> ' .. lr_file)
		end
		if savebaselineweight == 1 then
			torch.save('weights/1_conv1.t7', model:get(1):get(1).weight)
			for k=2,4 do
			   torch.save('weights/'..k..'_conv1.t7', model:get(1):get(k):get(1):get(3):get(1):get(1).weight)
			   torch.save('weights/'..k..'_conv2.t7', model:get(1):get(k):get(1):get(3):get(1):get(4).weight)
			   torch.save('weights/'..k..'_conv3.t7', model:get(1):get(k):get(1):get(3):get(2).weight)
			   torch.save('weights/'..k..'_conv4.t7', model:get(1):get(k):get(2):get(1):get(1):get(3).weight)
			   torch.save('weights/'..k..'_conv5.t7', model:get(1):get(k):get(2):get(1):get(1):get(6).weight)
			end
		end

	end

	--local final_loss = 0.001
	function getReward(batch_loss)
		local reward = 0
		--TODO: should get current error
		if batch_loss then
			--reward = 1 / math.abs(batch_loss - final_loss)
			if last_loss then
				log_sum = log_sum + math.log(batch_loss)-math.log(last_loss)
				assert(step_num >= 2, 'step_num should begin from 2 !')
				reward = -1/(step_num/10) * log_sum
			end
			last_loss = batch_loss
		end
		if verbose then
			if batch_loss and reward then
				print ('batch_loss: ' .. batch_loss)
				print ('reward: '.. reward)
			end
		end
		return reward
	end

	function getState(batch_loss) --state is set in cnn.lua

		--21*25 = 525
		--s1 = torch.mean(s1, 1):view(-1)
		--s2 = torch.mean(s2, 1):view(-1)
		--local tstate = torch.cat(s1, s2)
		--local s1 = model:get(1):get(1):get(2):get(1):get(1):get(6).weight
		local s1 = model:get(1):get(1).weight  --16,3,3,3
		s1 = s1:reshape(s1:size(1), s1:size(2), s1:size(3)*s1:size(4))

		function get_g_c(m)
			local r = m:reshape(m:nElement()) --m:view(-1)
			local r_sort = torch.sort(r)
			local n = r:nElement()
			local n1 = math.floor(n*0.25)
			local n2 = math.floor(n*0.5)
			local n3 = math.floor(n*0.75)-- quantiles(0.25, 0.5, 0.75)
			local g_c = torch.FloatTensor(12)
			g_c[1] = torch.mean(r)
			g_c[2] = r_sort[n1]
			g_c[3] = r_sort[n2]
			g_c[4] = r_sort[n3]
			g_c[5] = torch.std(r)
			g_c[6] = skewness(r)
			g_c[7] = kurtosis(r)
			g_c[8] = central_moment(r, 1)
			g_c[9] = central_moment(r, 2)
			g_c[10] = central_moment(r, 3)
			g_c[11] = central_moment(r, 4)
			g_c[12] = central_moment(r, 5)
			--local g_c_44 = torch.cat(g_c, k_bins_entropy(r))
			--return g_c_44
			return g_c
		end
		function get_h_d(s_param, type)
			--g_c
			local s = torch.Tensor(16,3,9):copy(s_param)
			local row = s:size(1)
			local col = s:size(2)
			local size = row
			type = type or 0
			if type == 1 then
				size = row * col
				s = s:reshape(size, s:size(3))
			end
			local g = torch.FloatTensor(size, 12)
			for i = 1, size do
				local g_c = get_g_c(s[i])
				g[i] = g_c
			end
			g = g:transpose(1, 2)  -- 13 rows
			--h_c
			local h = torch.FloatTensor(12, 5)
			for i = 1, 12 do
				local h_d = torch.FloatTensor(5)
				h_d[1] = torch.mean(g[i])
				h_d[2] = torch.median(g[i])
				h_d[3] = torch.std(g[i])
				h_d[4] = torch.max(g[i])
				h_d[5] = torch.min(g[i])
				h[i] = h_d
			end
			return h
		end

		local res = {}
		res[#res+1] = get_g_c(s1):view(-1)
		res[#res+1] = get_h_d(s1):view(-1)
		res[#res+1] = get_h_d(s1:transpose(1,2)):view(-1)
		res[#res+1] = get_h_d(s1, 1):view(-1)
		--concat LR
		res[#res+1] = torch.FloatTensor{cnnopt.learningRate}
		local state = res[1]
		for i = 2, #res do
			state = torch.cat(state, res[i])
		end
		--12 + 12*5 + 12*5 + 12*5 + 1 = 193

		local reward = getReward(batch_loss)
		if terminal == true then
			terminal = false
			return state, reward, true
		else
			return state, reward, false
		end
	end


	function step(state, batch_loss, action, tof)
		--take action from DQN, tune learning rate
		--TODO
		--[[
			action 1: decrease by 10%
			action 2: decrease by 50%
			action 3: restart
		]]
		step_num = step_num + 1
			local maxlearningRate = 0.5
			local minlearningRate = 0.001
			local learningRate_delta = cnnopt.learningRate --0.001 --opt.learningRate * 0.1
			if action == 1 then
				cnnopt.learningRate = cnnopt.learningRate - learningRate_delta*0.1
			elseif action == 2 then
				cnnopt.learningRate = cnnopt.learningRate - learningRate_delta*0.5
			elseif action == 3 then
				cnnopt.learningRate = 0.5
			end
			if cnnopt.learningRate > maxlearningRate then cnnopt.learningRate = maxlearningRate end
			if cnnopt.learningRate < minlearningRate then cnnopt.learningRate = minlearningRate end
			if verbose then
				print('action = ' .. action)
				print('learningRate = '..cnnopt.learningRate)
			end
			state.lr = cnnopt.learningRate
			os.execute('echo ' .. state.lr .. ' >> ' .. lr_file)
			state.config = tablex.deepcopy(cnnopt)
			state.optim = tablex.deepcopy(cnnopt)
		return getState(batch_loss)
	end

	if take_action == 1 then
		--DQN init
		screen, reward, terminal = getState(2.33)
		step_num = 1
	end

	local iteration_index = 0
	--will be called after each iteration
	engine.hooks.onForwardCriterion = function(state)
	    meter:add(state.criterion.output)
	    clerr:add(state.network.output, state.sample.target)
		if curr_mode == 'testcnn' then return end
			if take_action == 0 then return end
			local batch_loss = state.criterion.output
			iteration_index = iteration_index + 1
			if iteration_index % take_action_interval ~= 0 then
				return
			end
			--if iteration_index < 1/meta_momentum_coefficient and add_momentum == 1 then
			--  add_momentum_to_all_layer(model, tw)
			--end
			--given state, take action
			if verbose then
				print('--------------------------------------------------------')
			end
			local action_index = 0
			if episode % 2 == 1 then
			   action_index = agent:perceive(reward, screen, terminal)
			else
			   action_index = agent:perceive(reward, screen, terminal, true, 0.05)
			end
			if not terminal then
			   screen, reward, terminal = step(state, batch_loss, game_actions[action_index], true)
			else
			   screen, reward, terminal = getState(batch_loss)
			   reward = 0
			   terminal = false
			end
		end

	local inputs = cast(torch.Tensor())
	local targets = cast(torch.Tensor())
	engine.hooks.onSample = function(state)
		inputs:resize(state.sample.input:size()):copy(state.sample.input)
		targets:resize(state.sample.target:size()):copy(state.sample.target)
		state.sample.input = inputs
		state.sample.target = targets
	end

	engine:train{
		network = model,
		iterator = getIterator('train'),
		criterion = criterion,
		optimMethod = optim.sgd,
		config = tablex.deepcopy(cnnopt),
		maxepoch = cnnopt.max_epoch,
	}



end
