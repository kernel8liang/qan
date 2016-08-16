--[[
Copyright (c) 2014 Google Inc.

See LICENSE file for full terms of limited license.
]]

if not dqn then
    require "initenv"
end

local cmd = torch.CmdLine()
cmd:text()
cmd:text('Train Agent in Environment:')
cmd:text()
cmd:text('Options:')

cmd:option('-framework', '', 'name of training framework')
cmd:option('-env', '', 'name of environment to use')
cmd:option('-game_path', '', 'path to environment file (ROM)')
cmd:option('-env_params', '', 'string of environment parameters')
cmd:option('-pool_frms', '',
           'string of frame pooling parameters (e.g.: size=2,type="max")')
cmd:option('-actrep', 1, 'how many times to repeat action')
cmd:option('-random_starts', 0, 'play action 0 between 1 and random_starts ' ..
           'number of times at the start of each training episode')

cmd:option('-name', '', 'filename used for saving network and training history')
cmd:option('-network', '', 'reload pretrained network')
cmd:option('-agent', '', 'name of agent file to use')
cmd:option('-agent_params', '', 'string of agent parameters')
cmd:option('-seed', 1, 'fixed input seed for repeatable experiments')
cmd:option('-saveNetworkParams', false,
           'saves the agent network in a separate file')
cmd:option('-prog_freq', 5*10^7, 'frequency of progress output')
cmd:option('-save_freq', 5*10^7, 'the model is saved every save_freq steps')
cmd:option('-eval_freq', 10^7, 'frequency of greedy evaluation')
cmd:option('-save_versions', 0, '')

cmd:option('-steps', 10^6, 'number of training steps to perform')
cmd:option('-eval_steps', 10^5, 'number of evaluation steps')

cmd:option('-verbose', 2,
           'the higher the level, the more information is printed to screen')
cmd:option('-threads', 1, 'number of BLAS threads')
cmd:option('-gpu', -1, 'gpu flag')
cmd:option('-max_episode', 50, '')
cmd:option('-take_action', 1, '')
cmd:option('-savebaselineweight', 0, '')
cmd:option('-output_file', 'logs/torchnet_test_loss.log', '')

cmd:text()

local opt = cmd:parse(arg)

--- General setup.
--local game_env, game_actions, agent, opt = setup(opt)
local game_actions, agent, opt = setup(opt)

-- override print to always flush the output
local old_print = print
local print = function(...)
    old_print(...)
    io.flush()
end

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

local total_reward
local nrewards
local nepisodes
local episode_reward


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


max_episode = opt.max_episode or 50
output_file = opt.output_file or 'logs/torchnet_test_loss.log'
take_action = opt.take_action or 1
savebaselineweight = opt.savebaselineweight or 0
validation_output_file = 'logs/validation_loss.log'
local episode = 0
os.execute('rm -f ' .. output_file)
os.execute('rm -f ' .. validation_output_file)
os.execute('mkdir weights')

while episode < max_episode do
		episode = episode + 1
		local last_validation_loss = 10000
		local early_stop = false
		local opt = {
		  dataset = './datasets/trainvalidata.t7',
		  save = 'logs',
		  batchSize = 128,
		  learningRate = 0.1,
		  learningRateDecay = 0,
		  learningRateDecayRatio = 0.2,
		  weightDecay = 0.0005,
		  dampening = 0,
		  momentum = 0.9,
		  epoch_step = "80",
		  max_epoch = 3,
		  model = 'nin',
		  optimMethod = 'sgd',
		  init_value = 10,
		  depth = 50,
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
		  data_type = 'torch.CudaTensor',
		}
		opt = xlua.envparams(opt)

		opt.epoch_step = tonumber(opt.epoch_step) or loadstring('return '..opt.epoch_step)()
		print(opt)

		print(c.blue '==>' ..' loading data')
		local provider = torch.load(opt.dataset)
		opt.num_classes = provider.testData.labels:max()

		local function cast(x) return x:type(opt.data_type) end

		print(c.blue '==>' ..' configuring model')
		local model = nn.Sequential()
		local net = dofile('models/'..opt.model..'.lua')(opt)
		if opt.data_type:match'torch.Cuda.*Tensor' then
		   require 'cudnn'
		   require 'cunn'
		   cudnn.convert(net, cudnn):cuda()
		   cudnn.benchmark = true
		   if opt.cudnn_deterministic then
			  net:apply(function(m) if m.setMode then m:setMode(1,1,1) end end)
		   end

		   print(net)
		   print('Network has', #net:findModules'cudnn.SpatialConvolution', 'convolutions')

		   local sample_input = torch.randn(8,3,opt.imageSize,opt.imageSize):cuda()
		   if opt.generate_graph then
			  iterm.dot(graphgen(net, sample_input), opt.save..'/graph.pdf')
		   end
		   if opt.optnet_optimize then
			  optnet.optimizeMemory(net, sample_input, {inplace = false, mode = 'training'})
		   end

		end
		model:add(utils.makeDataParallelTable(net, opt.nGPU))
		cast(model)

		local function hflip(x)
		   return torch.random(0,1) == 1 and x or image.hflip(x)
		end

		local function randomcrop(x)
		   local pad = opt.randomcrop
		   if opt.randomcrop_type == 'reflection' then
			  module = nn.SpatialReflectionPadding(pad,pad,pad,pad):float()
		   elseif opt.randomcrop_type == 'zero' then
			  module = nn.SpatialZeroPadding(pad,pad,pad,pad):float()
		   else
			  error'unknown mode'
		   end

		   local imsize = opt.imageSize
		   local padded = module:forward(x)
		   local x = torch.random(1,pad*2 + 1)
		   local y = torch.random(1,pad*2 + 1)
		   return padded:narrow(3,x,imsize):narrow(2,y,imsize)
		end


		local function getIterator(mode)
		   return tnt.ParallelDatasetIterator{
			  nthread = 8,
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
							 opt.hflip and hflip,
							 opt.randomcrop > 0 and randomcrop,
						  }
					   }
					   :batch(opt.batchSize, 'skip-last')
				 elseif mode == 'test' then
					return list_dataset
					   :batch(opt.batchSize, 'include-last')
				 elseif mode == 'validation' then
                    return list_dataset
                       :batch(opt.batchSize, 'include-last')
                 end
			  end
		   }
		end

		local function log(t) print('json_stats: '..json.encode(tablex.merge(t,opt,true))) end

		print('Will save at '..opt.save)
		paths.mkdir(opt.save)

		local engine = tnt.OptimEngine()
		local criterion = cast(nn.CrossEntropyCriterion())
		local meter = tnt.AverageValueMeter()
		local clerr = tnt.ClassErrorMeter{topk = {1}}
		local train_timer = torch.Timer()
		local test_timer = torch.Timer()

		engine.hooks.onStartEpoch = function(state)
		   local epoch = state.epoch + 1
		   print('==>'.." online epoch # " .. epoch .. ' [batchSize = ' .. opt.batchSize .. ']')
		   meter:reset()
		   clerr:reset()
		   train_timer:reset()
		   if torch.type(opt.epoch_step) == 'number' and epoch % opt.epoch_step == 0 or
			  torch.type(opt.epoch_step) == 'table' and tablex.find(opt.epoch_step, epoch) then
			  opt.learningRate = opt.learningRate * opt.learningRateDecayRatio
			  state.config = tablex.deepcopy(opt)
			  state.optim = tablex.deepcopy(opt)
		   end
		end

		engine.hooks.onEndEpoch = function(state)
		   local train_loss = meter:value()
		   local train_err = clerr:value{k = 1}
		   local train_time = train_timer:time().real
		   meter:reset()
		   clerr:reset()
		   test_timer:reset()

		   engine:test{
			  network = model,
			  iterator = getIterator('test'),
			  criterion = criterion,
		   }

		   log{
			  loss = train_loss,
			  train_loss = train_loss,
			  train_acc = 100 - train_err,
			  epoch = state.epoch,
			  test_acc = 100 - clerr:value{k = 1},
			  lr = opt.learningRate,
			  train_time = train_time,
			  test_time = test_timer:time().real,
			  n_parameters = state.params:numel(),
		   }
		   os.execute('echo ' .. (100 - clerr:value{k = 1}) .. '>> ' .. output_file)

		   meter:reset()
		   clerr:reset()
		   test_timer:reset()

		   engine:test{
			   network = model,
			   iterator = getIterator('validation'),
			   criterion = criterion,
		   }
		   os.execute('echo ' .. clerr:value{k = 1 } .. ' >> ' .. validation_output_file)
		   if state.epoch > 28 and clerr:value{k = 1 } > last_validation_loss then
			   early_stop = true
			   state.epoch = opt.max_epoch
			   os.execute('echo "episode_end" >> ' .. output_file)
			   os.execute('echo "episode_end" >> ' .. validation_output_file)
		   end
		   last_validation_loss = clerr:value{k = 1 }
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

        print (savebaselineweight)
        print (max_episode)
        if take_action == 1 then
		    baseline_weights = torch.load('weights/4_conv5.t7') --the top conv layer
        end

		function getState(batch_loss, verbose) --state is set in cnn.lua
			verbose = verbose or false
			--return state, reward, term
			--print(self.model:get(3):get(54):get(6))
			local k = 4
			local v = {}
			for i=1,16 do v[i]=i end
			local tstate = model:get(1):get(k):get(2):get(1):get(1):get(6).weight:index(1, torch.LongTensor(v))
			--get model weight as state --model:get(2):get(1).weight:view(64,27):index(1, torch.LongTensor(v))
			--local tstate = self.model:get(2):get(54):get(6).weight 
			print(tstate:size())
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
			local maxlearningRate = 1
			local minlearningRate = 0.01
			local learningRate_delta = 0.01 --opt.learningRate * 0.1
			if action == 1 then
				opt.learningRate = opt.learningRate + learningRate_delta
			elseif action == 2 then
				opt.learningRate = opt.learningRate - learningRate_delta
			end
			if opt.learningRate > maxlearningRate then opt.learningRate = maxlearningRate end
			if opt.learningRate < minlearningRate then opt.learningRate = minlearningRate end
			print('learningRate = '..opt.learningRate)
			state.config = tablex.deepcopy(opt)
			state.optim = tablex.deepcopy(opt)
			return getState(batch_loss, true)
		end

        if take_action == 1 then
            --DQN init
            screen, reward, terminal = getState(2.33, true)
            meta_momentum_coefficient = 0.0001
            tw = {}
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
		function meta_momentum(w, targetw)
			local tmp = torch.CudaTensor(targetw:size()):copy(targetw)
			w:add(tmp:add(-w):mul(meta_momentum_coefficient))  --w = w + (target_w - w) * co-efficient
		end

		local iteration_index = 0
		--will be called after each iteration
		engine.hooks.onForwardCriterion = function(state)
		   meter:add(state.criterion.output)
		   clerr:add(state.network.output, state.sample.target)
		   if take_action == 0 then return end
		   local batch_loss = state.criterion.output
		   iteration_index = iteration_index + 1

		   if iteration_index < 10000 then
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
		   --given state, take action
		   print('--------------------------------------------------------')
		   local action_index = agent:perceive(reward, screen, terminal)
		   if not terminal then
			   screen, reward, terminal = step(state, batch_loss, game_actions[action_index], true)
		   else
			   screen, reward, terminal = getState(batch_loss, true)
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
		   config = tablex.deepcopy(opt),
		   maxepoch = opt.max_epoch,
		}

		torch.save(opt.save..'/model.t7', net:clearState())

end
