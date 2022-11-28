# Superparameters
OUTPUT_GRAPH = True
MAX_EPISODE = 100    ## i presume as epoch coz no batch  ## finish whole ML Sim
DISPLAY_REWARD_THRESHOLD = 200  # renders environment if total episode reward is greater then this threshold
MAX_EP_STEPS = 90   # maximum time step in one episode
# RENDER = False  # rendering wastes time
GAMMA = 0.99     # reward discount in TD error
LR = 0.0001    # learning rate 
BATCH_SIZE = 64   # same to Alibaba paper <<A cold-start-free reinforcement learning approach for traffic signal control>>
REPLAY_MEM_SIZE = 20000
EPOCH = 1050  ## finish all batch one epoch
# EPISODES = 10
RANDOMSEED = True
SAVE_MODEL = True

LEAKY_RELU_ALPHA = 0.01