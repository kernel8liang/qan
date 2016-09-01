--[[
Copyright (c) 2014 Google Inc.

See LICENSE file for full terms of limited license.
]]

if not dqn then require "initenv" end
require 'config'
require 'functions'
--CNN setting
require 'xlua'
require 'optim'
require 'image'
local tnt = require 'torchnet'
local c = require 'trepl.colorize'
local json = require 'cjson'
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

local usegpu = true

if take_action == 1 and add_momentum == 1 then
	tw = {}
	loadbaseweight(tw)
end
while episode < max_episode do
	--collectgarbage()
	episode = episode + 1
	local last_validation_loss = 10000
	local early_stop = false
	local min_epoch = 10
	local cnnopt = {
		learningRate = 0.005
	}
	local function getIterator(mode)
		return tnt.ParallelDatasetIterator{
			nthread = 1,
			init    = function() require 'torchnet' end,
			closure = function()

				-- load MNIST dataset:
				local mnist = require 'mnist'
				local dataset = mnist[mode .. 'dataset']()

				dataset.data = dataset.data:reshape(dataset.data:size(1),
					dataset.data:size(2) * dataset.data:size(3)):double()
				-- return batches of data:
				return tnt.BatchDataset{
					batchsize = 128,
					dataset = tnt.ListDataset{  -- replace this by your own dataset
						list = torch.range(1, dataset.data:size(1)):long(),
						load = function(idx)
							return {
								input  = dataset.data[idx],
								target = torch.LongTensor{dataset.label[idx] + 1},
							}  -- sample contains input and target
						end,
					}
				}
			end,
		}
	end

	-- set up logistic regressor:
	--[[
	local net = nn.Sequential()
	local Convolution = nn.SpatialConvolution
	local Max = nn.SpatialMaxPooling
	local Linear = nn.Linear
	local Tanh = nn.Tanh
	local Reshape = nn.Reshape
	net:add(Reshape(1,28,28))
	net:add(Convolution(1,20,5,5))
	net:add(nn.Tanh())
	net:add(Max(2,2,2,2))
	net:add(Convolution(20,50,5,5))
	net:add(nn.Tanh())
	net:add(Max(2,2,2,2))
	net:add(Reshape(50*4*4))
	net:add(Linear(50*4*4, 500))
	net:add(nn.Tanh())
	net:add(Linear(500, 10))
	]]
	local net = torch.load('weights/net9.t7')
	for i=1,8 do
		if net:get(i).weight then
			print(net:get(i).weight:size())
		end
	end
	print(net)
	local criterion = nn.CrossEntropyCriterion()

	-- set up training engine:
	local engine = tnt.SGDEngine()
	local meter  = tnt.AverageValueMeter()
	local clerr  = tnt.ClassErrorMeter{topk = {1}}
	engine.hooks.onStartEpoch = function(state)
		meter:reset()
		clerr:reset()
		print('start epoch')
	end
	engine.hooks.onForwardCriterion = function(state)
		meter:add(state.criterion.output)
		clerr:add(state.network.output, state.sample.target)
		if state.training then
			print(string.format('avg. loss: %2.4f; avg. error: %2.4f',
				meter:value(), clerr:value{k = 1}))
		end
	end

	-- set up GPU training:
	if usegpu then
		-- copy model to GPU:
		require 'cunn'
		net       = net:cuda()
		criterion = criterion:cuda()
		-- copy sample to GPU buffer:
		local igpu, tgpu = torch.CudaTensor(), torch.CudaTensor()
		engine.hooks.onSample = function(state)
			igpu:resize(state.sample.input:size() ):copy(state.sample.input)
			tgpu:resize(state.sample.target:size()):copy(state.sample.target)
			state.sample.input  = igpu
			state.sample.target = tgpu
		end  -- alternatively, this logic can be implemented via a TransformDataset
	end


	engine.hooks.onEndEpoch = function(state)
		local train_loss = meter:value()
		local train_err = clerr:value{k = 1}
		meter:reset()
		clerr:reset()
		curr_mode = 'testcnn'
		engine:test{
			network = net,
			iterator = getIterator('test'),
			criterion = criterion,
		}
		curr_mode = 'traincnn'


		local acc = 100 - clerr:value{k = 1}
		print('acc = ' .. acc)
		os.execute('echo ' .. (100 - clerr:value{k = 1}) .. '>> ' .. output_file)
		if state.epoch == state.maxepoch then
			os.execute('echo ------ >> ' .. output_file)
			os.execute('echo ------ >> ' .. lr_file)
		end
		if savebaselineweight == 1 then
			--torch.save('weights/w2.t7', net:get(2).weight)
			--torch.save('weights/w5.t7', net:get(5).weight)
			--torch.save('weights/w9.t7', net:get(9).weight)
			torch.save('weights/net' .. state.epoch .. '.t7', net)
		end

	end

	local final_loss = 0.0001
	function getReward(batch_loss, verbose)
		verbose = verbose or false
		local reward = 0
		--TODO: should get current error
		if batch_loss then
			reward = 1 / math.abs(batch_loss - final_loss)
		end
		if (verbose) then
			print ('final_loss: ' .. final_loss)
			if batch_loss then print ('batch_loss: ' .. batch_loss) end
			print ('reward: '.. reward)
		end
		return reward
	end

	if take_action == 1 then
		baseline_weights = torch.load('weights/w.t7') --the top conv layer
	end

	function getState(batch_loss, verbose) --state is set in cnn.lua
		verbose = verbose or false
		local v = {}
		for i=1,16 do v[i]=i end
		local tstate = net:get(5).weight
		local reward = getReward(batch_loss, verbose)
		if terminal == true then
			terminal = false
			return tstate, reward, true
		else
			return tstate, reward, false
		end
	end

	function step(state, batch_loss, action, tof)
		--take action from DQN, tune learning rate
		--TODO
		--[[
			action 1: increase
			action 2: decrease
			action 3: unchanged
		]]
		local maxlearningRate = 0.05
		local minlearningRate = 0.001
		local learningRate_delta = 0.001 --opt.learningRate * 0.1
		print('action = ' .. action)
		if action == 1 then
			cnnopt.learningRate = cnnopt.learningRate + learningRate_delta
		elseif action == 2 then
			cnnopt.learningRate = cnnopt.learningRate - learningRate_delta
		end
		if cnnopt.learningRate > maxlearningRate then cnnopt.learningRate = maxlearningRate end
		if cnnopt.learningRate < minlearningRate then cnnopt.learningRate = minlearningRate end
		print('learningRate = '..cnnopt.learningRate)
		state.lr = cnnopt.learningRate
		os.execute('echo ' .. state.lr .. ' >> ' .. lr_file)
		return getState(batch_loss, true)
	end
	if take_action == 1 then
		--DQN init
		screen, reward, terminal = getState(2.33, true)
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
		if iteration_index < momentum_times and add_momentum == 1 then
			add_momentum_to_all_layer(net, tw)
		end
		--given state, take action
		print('--------------------------------------------------------')
		local action_index = 0
		if episode % 2 == 1 then
		   action_index = agent:perceive(reward, screen, terminal)
		else
		   action_index = agent:perceive(reward, screen, terminal, true, 0.05)
		end
		if not terminal then
		   screen, reward, terminal = step(state, batch_loss, game_actions[action_index], true)
		else
		   screen, reward, terminal = getState(batch_loss, true)
		   reward = 0
		   terminal = false
		end
	end

	-- train the model:
	engine:train{
		network   = net,
		iterator  = getIterator('train'),
		criterion = criterion,
		lr = cnnopt.learningRate,
		maxepoch = 5
		--optimMethod = optim.sgd,
		--config = tablex.deepcopy(cnnopt),
		--maxepoch = cnnopt.max_epoch,
	}

	local ave_q_max = 0
	ave_q_max = agent:getAveQ()
	print('ave_q_max = ')
	print(ave_q_max)
	print('Q file = ')
	print(Q_file)
	os.execute('echo ' .. ave_q_max .. ' >> ' .. Q_file)

end
