
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
cmd:option('-max_episode', 500, '')
cmd:option('-take_action', 1, '')
cmd:option('-savebaselineweight', 0, '')
cmd:option('-output_file', 'logs/torchnet_test_loss.log', '')

cmd:text()
opt = cmd:parse(arg)

max_episode = 10
output_file = 'logs/torchnet_test_loss.log'
train_output_file = 'logs/torchnet_train_loss.log'
validation_output_file = 'logs/validation_loss.log'
lr_file = 'logs/learning_rate.log'
Q_file = 'logs/Q.log'
take_action = 0
savebaselineweight = 0
curr_mode = 'traincnn'

episode = 0
os.execute('rm -f ' .. output_file)
os.execute('rm -f ' .. validation_output_file)
os.execute('rm -f ' .. train_output_file)
os.execute('rm -f ' .. Q_file)
os.execute('mkdir weights')


--[[
function getopt()
    local replay_memory=1000000
    local eps_end=0.1
    local eps_endt=replay_memory
    local lr=0.00025
    local discount=0.99
    local learn_start=50000
    local update_freq=4
    local n_replay=1
    local preproc_net="\"net_downsample_2x_full_y\""
    local netfile="\"convnet_atari3\""
    local state_dim=7056
    local ncols=1
    local pool_frms_type="\"max\""
    local pool_frms_size=2
    local opt = {
        framework = nil,
        game_path=nil,
        env_params=nil,
        agent="NeuralQLearner",
        actrep=4,
        seed=1,
        initial_priority="false",
        agent_params="lr="..lr..",ep=1,ep_end="..eps_end..",ep_endt="..eps_endt..",discount="..discount..",hist_len=4,learn_start="..learn_start..",replay_memory="..replay_memory..",update_freq="..update_freq..",n_replay="..n_replay..",network="..netfile..",preproc="..preproc_net..",state_dim="..state_dim..",minibatch_size=32,rescale_r=1,ncols="..ncols..",bufferSize=512,valid_size=500,target_q=10000,clip_delta=1,min_reward=-1,max_reward=1",
        steps=50000000,
        eval_freq=25000000,
        eval_steps=12500000,
        prog_freq=10000000,
        save_freq=12500000,
        gpu=0,
        random_starts=30,
        pool_frms="type="..pool_frms_type..",size="..pool_frms_size,
        num_threads=4
    }
end
]]
