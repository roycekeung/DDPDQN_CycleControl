

import wandb
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Dropout, Activation, Flatten, \
    Conv2D, MaxPool2D, LeakyReLU, Add

from tensorflow_addons.layers import NoisyDense 
import pickle


import numpy as np
from collections import deque

from config import *

from DQNtype import AlgoType


__all__ = [ 'NetModel', 'AgentController', 'collections']  


'''
eg.
discrete actions

the actions’ space is defined by how to update
the duration of every phase in the next cycle. Considering the
system may become unstable if the duration change between
two cycles is too large, we specify a change step. In this paper,
we set it to be 5 seconds. We model the duration changes of two
phases between two neighboring cycles as a high-dimension
MDP. In the model, the traffic light changes only one phase’s
duration by 5 seconds if there is any change

'''

## seed
np.random.seed(np.random.randint(1000) if RANDOMSEED else 1)
tf.random.set_seed(np.random.randint(1000) if RANDOMSEED else 1)   ## tf2.0 


class SumTree(object):
    """
    This SumTree code is a modified version and the original code is from:
    https://github.com/jaara/AI-blog/blob/master/SumTree.py

    Story data with its priority in the tree.
    """
    data_pointer = 0

    def __init__(self, capacity):
        self.capacity = capacity  # for all priority values
        self.tree = np.zeros(2 * capacity - 1)
        # ? --> see Tree structure annotation in get_leaf(); assume the total (weights per batch) number is even
        # [--------------Parent nodes-------------][-------leaves to recode priority-------]
        #             size: capacity - 1                       size: capacity
        self.data = np.zeros(capacity, dtype=object)  # for all transitions
        ## data = transition = ((s, [a, r], s_))
        # [--------------data frame-------------]
        #             size: capacity
    #### later on finish it; this tree is very complicated


class ExperienceReplay(object):
    '''Experience replay <st,at,rt,st+1> '''
    def __init__(self, memory_size:int=10000 ):
        ## limit capacity itself
        self.memory = deque(maxlen= memory_size)
        
    def sampling(self, batch_size= 32):
        ## could select the same index multiple times
        sample_index = np.random.choice(self.get_size(), size=batch_size , replace=True)
        batch_memory = []
        for x in sample_index:
            batch_memory.append(self.memory[x])
            
        states, actions, rewards, next_states, is_done = map(np.asarray, zip(*batch_memory))
        states = np.array(states).reshape(batch_size, -1)  ## row of each state_t
        next_states = np.array(next_states).reshape(batch_size, -1)  ## row of each next_state_t
        
        return states, actions, rewards, next_states, is_done
        
    def store_transition(self, state, action, reward, next_state, is_done):
        ## <st,at,rt,st+1> 
        self.memory.append([state, action, reward, next_state, is_done])
    
    def get_size(self):
        return len(self.memory)
    
    def get_mem(self):
        return self.memory
    
    def clear(self):
        self.memory.clear()
        self.memory_counter = 0



class NetModel(object):
    ## value-based off-policy
    def __init__(self, feature_shape:tuple , n_actions:int, e_greedy:float=0.9, e_decay:float=0.99, e_min:float=0, tensorboard=None, algo_type=AlgoType.Noraml, isbuild_net=True ):
        '''
        feature_shape = (whatever x size , whatever y size, 1 or 2(pos and speed ) or x 4  coz frames of history)
        The number of grids at an intersection is 60 × 60
        The input data become 60 × 60 × 2 with both position and speed information ;  feature_shape(60 , 60, 2)
        image
        '''
        self.feature_shape = feature_shape
        self.n_actions = n_actions
        self.epsilon = e_greedy
        self.epsilon_decay = e_decay
        self.epsilon_min = e_min
        self.tensorboard = tensorboard
        self.isbuild_net= isbuild_net
        if isbuild_net:
            ### timecost-effective Adam, Nadam, RMSProp, and Adamax
            ### if have time , search deeply : Stochastic Gradient Descent 
            self.optimizer = tf.keras.optimizers.Adam(LR)      ### tune alpha instead of epsilon, beta1 and beta2
            self.model = self.build_net()
        else:
            self.model = None

    def build_net(self, algo_type)->tf.keras.Model:
        if algo_type==AlgoType.Noraml or algo_type== AlgoType.Double:
            ### Normal DQN and Double DQN
            model = tf.keras.models.Sequential()
            # inn = Input(shape=(feature_shape))
            '''
            The network consists a stack of two convolutional layers with 
            32 filters(channels), 4 × 4 kernel, 2 × 2 strides with padding 0 replacement;
            64 filters(channels), 2 × 2 kernel, 2 × 2 strides with padding 0 replacement;
            128 filters(channels), 2 × 2 kernel, 1 × 1 strides with padding 0 replacement
            ...
             (same to atari<<Playing Atari with Deep Reinforcement Learning>>)
            '''
            model.add(Conv2D(filters = 32, 
                            kernel_size=(4, 4), 
                            strides=(2,2), 
                            padding='same', 
                            kernel_initializer=tf.random_normal_initializer(0., .1), 
                            bias_initializer=tf.constant_initializer(0.1), 
                            input_shape=(self.feature_shape) ,
                            kernel_regularizer=tf.keras.regularizers.L2(0.01),  ## on w , not on b a; w->0; penalize high w 
                            kernel_constraint=tf.keras.constraints.MaxNorm(3),   ## better with L2 as limit
                            name="layer1",
                            )
                    )
            model.add(LeakyReLU(alpha=LEAKY_RELU_ALPHA))
            ## model.add(MaxPooling2D(pool_size=(2, 2)))   ### ??? paper didnt sxplain it clearly
            model.add(Conv2D(filters = 64, 
                            kernel_size=(2, 2), 
                            strides=(2,2), 
                            padding='same',
                            kernel_initializer=tf.random_normal_initializer(0., .1), 
                            bias_initializer=tf.constant_initializer(0.1), 
                            kernel_regularizer=tf.keras.regularizers.L2(0.01),  ## on w , not on b a; w->0; penalize high w 
                            kernel_constraint=tf.keras.constraints.MaxNorm(3),   ## better with L2 as limit
                            name="layer2" , 
                            )
                    )
            model.add(LeakyReLU(alpha=LEAKY_RELU_ALPHA))
            ## model.add(MaxPooling2D(pool_size=(2, 2)))   ### ??? paper didnt sxplain it clearly
            model.add(Conv2D(filters = 128, 
                            kernel_size=(2, 2), 
                            strides=(1,1), 
                            padding='same', 
                            kernel_initializer=tf.random_normal_initializer(0., .1), 
                            bias_initializer=tf.constant_initializer(0.1), 
                            kernel_regularizer=tf.keras.regularizers.L2(0.01),  ## on w , not on b a; w->0; penalize high w 
                            kernel_constraint=tf.keras.constraints.MaxNorm(3),   ## better with L2 as limit
                            name="layer3", 
                            )
                    )
            model.add(LeakyReLU(alpha=LEAKY_RELU_ALPHA))
            ## model.add(MaxPooling2D(pool_size=(2, 2)))   ### ??? not mentioned
            model.add(Flatten())
            ### i amend it ; paper dont have these option just Normal DQN and Double DQN
            # model.add(Dense(128, kernel_initializer=tf.random_normal_initializer(0., .1), bias_initializer=tf.constant_initializer(0.1), name="layer4"))
            model.add(NoisyDense(128, 
                                 sigma = 0.5, 
                                 use_factorised = True, 
                                 kernel_initializer=tf.random_normal_initializer(0., .1), 
                                 bias_initializer=tf.constant_initializer(0.1), 
                                 kernel_regularizer=tf.keras.regularizers.L2(0.01),  ## on w , not on b a; w->0; penalize high w 
                                 kernel_constraint=tf.keras.constraints.MaxNorm(3),   ## better with L2 as limit
                                 name="layer4"))  ## noisy net factorised Gaussian noise
            model.add(LeakyReLU(alpha=LEAKY_RELU_ALPHA))
            # model.add(Dense(64), kernel_initializer=tf.random_normal_initializer(0., .1), bias_initializer=tf.constant_initializer(0.1), name="layer5")
            model.add(NoisyDense(64, 
                                 sigma = 0.5, 
                                 use_factorised = True, 
                                 kernel_initializer=tf.random_normal_initializer(0., .1), 
                                 bias_initializer=tf.constant_initializer(0.1), 
                                 kernel_regularizer=tf.keras.regularizers.L2(0.01),  ## on w , not on b a; w->0; penalize high w 
                                 kernel_constraint=tf.keras.constraints.MaxNorm(3),   ## better with L2 as limit
                                 name="layer5"))  ## noisy net factorised Gaussian noise
            model.add(LeakyReLU(alpha=LEAKY_RELU_ALPHA))
            # model.add(Dense(32), kernel_initializer=tf.random_normal_initializer(0., .1), bias_initializer=tf.constant_initializer(0.1), name="layer6")
            model.add(NoisyDense(32, 
                                 sigma = 0.5, 
                                 use_factorised = True, 
                                 kernel_initializer=tf.random_normal_initializer(0., .1), 
                                 bias_initializer=tf.constant_initializer(0.1), 
                                 kernel_regularizer=tf.keras.regularizers.L2(0.01),  ## on w , not on b a; w->0; penalize high w 
                                 kernel_constraint=tf.keras.constraints.MaxNorm(3),   ## better with L2 as limit
                                 name="layer6"))  ## noisy net factorised Gaussian noise
            model.add(LeakyReLU(alpha=LEAKY_RELU_ALPHA))
            # model.add(Dense(self.n_actions, activation='linear'))
            model.add(NoisyDense(self.n_actions, 
                                 sigma = 0.5, 
                                 use_factorised = True, 
                                 activation='linear', 
                                 name='outputlayer'))  ## noisy net factorised Gaussian noise
            model.compile(loss='mse', optimizer=self.optimizer )  ## no metric coz no exact y_pred to compare
            return model
        elif algo_type== AlgoType.Dueling:
            ### Dueling DQN
            leaky_relu = LeakyReLU(alpha=LEAKY_RELU_ALPHA)
            
            state_input = Input(shape=self.feature_shape)
            '''
            The network consists a stack of two convolutional layers with 
            32 filters(channels), 4 × 4 kernel, 2 × 2 strides with padding 0 replacement;
            64 filters(channels), 2 × 2 kernel, 2 × 2 strides with padding 0 replacement;
            128 filters(channels), 2 × 2 kernel, 1 × 1 strides with padding 0 replacement
            ...
            (same to atari<<Playing Atari with Deep Reinforcement Learning>>)
            '''
            Covlayer1 = Conv2D(filters = 32, 
                            kernel_size=(4, 4), 
                            strides=(2,2), 
                            padding='same', 
                            activation = leaky_relu,
                            kernel_initializer=tf.random_normal_initializer(0., .1), 
                            bias_initializer=tf.constant_initializer(0.1), 
                            input_shape=(self.feature_shape) ,
                            kernel_regularizer=tf.keras.regularizers.L2(0.01),  ## on w , not on b a; w->0; penalize high w 
                            kernel_constraint=tf.keras.constraints.MaxNorm(3),   ## better with L2 as limit
                            name="layer1"
                            )(state_input)
            Covlayer2 = Conv2D(filters = 64, 
                            kernel_size=(2, 2), 
                            strides=(2,2), 
                            padding='same',
                            activation = leaky_relu,
                            kernel_initializer=tf.random_normal_initializer(0., .1), 
                            bias_initializer=tf.constant_initializer(0.1), 
                            kernel_regularizer=tf.keras.regularizers.L2(0.01),  ## on w , not on b a; w->0; penalize high w 
                            kernel_constraint=tf.keras.constraints.MaxNorm(3),   ## better with L2 as limit
                            name="layer2"
                            )(Covlayer1)
            Covlayer3 = Conv2D(filters = 128, 
                            kernel_size=(2, 2), 
                            strides=(1,1), 
                            padding='same', 
                            activation = leaky_relu,
                            kernel_initializer=tf.random_normal_initializer(0., .1), 
                            bias_initializer=tf.constant_initializer(0.1), 
                            kernel_regularizer=tf.keras.regularizers.L2(0.01),  ## on w , not on b a; w->0; penalize high w 
                            kernel_constraint=tf.keras.constraints.MaxNorm(3),   ## better with L2 as limit
                            name="layer3"
                            )(Covlayer2)
            flatlayer = Flatten()(Covlayer3)
            # flatlayer4 = Dense(128, activation = leaky_relu,  kernel_initializer=tf.random_normal_initializer(0., .1), bias_initializer=tf.constant_initializer(0.1), name="layer4")(flatlayer)
            flatlayer4 = NoisyDense(128, sigma = 0.5, 
                                    use_factorised = True, 
                                    activation = leaky_relu, 
                                    kernel_initializer=tf.random_normal_initializer(0., .1), 
                                    bias_initializer=tf.constant_initializer(0.1), 
                                    kernel_regularizer=tf.keras.regularizers.L2(0.01),  ## on w , not on b a; w->0; penalize high w 
                                    kernel_constraint=tf.keras.constraints.MaxNorm(3),   ## better with L2 as limit
                                    name="layer4")(flatlayer)  ## noisy net factorised Gaussian noise
            # flatlayer5_1 = Dense(64, activation = leaky_relu,  kernel_initializer=tf.random_normal_initializer(0., .1), bias_initializer=tf.constant_initializer(0.1), name="layer5_1")(flatlayer4)
            flatlayer5_1 = NoisyDense(64, 
                                      sigma = 0.5, 
                                      use_factorised = True, 
                                      activation = leaky_relu,  
                                      kernel_initializer=tf.random_normal_initializer(0., .1), 
                                      bias_initializer=tf.constant_initializer(0.1), 
                                      kernel_regularizer=tf.keras.regularizers.L2(0.01),  ## on w , not on b a; w->0; penalize high w 
                                      kernel_constraint=tf.keras.constraints.MaxNorm(3),   ## better with L2 as limit
                                      name="layer5_1")(flatlayer4)   ## noisy net factorised Gaussian noise
            # flatlayer5_2 = Dense(64, activation = leaky_relu,  kernel_initializer=tf.random_normal_initializer(0., .1), bias_initializer=tf.constant_initializer(0.1), name="layer5_2")(flatlayer4)
            flatlayer5_2 = NoisyDense(64, 
                                      sigma = 0.5, 
                                      use_factorised = True, 
                                      activation = leaky_relu,  
                                      kernel_initializer=tf.random_normal_initializer(0., .1), 
                                      bias_initializer=tf.constant_initializer(0.1), 
                                      kernel_regularizer=tf.keras.regularizers.L2(0.01),  ## on w , not on b a; w->0; penalize high w 
                                      kernel_constraint=tf.keras.constraints.MaxNorm(3),   ## better with L2 as limit
                                      name="layer5_2")(flatlayer4)   ## noisy net factorised Gaussian noise
            
            value_output = Dense(1, activation = 'linear', name="value_output_layer")(flatlayer5_1)     ## noisy net factorised Gaussian noise
            advantage_output = Dense(self.n_actions, activation = 'linear', name="advantage_output_layer")(flatlayer5_2)    ## noisy net factorised Gaussian noise
            ### tentative action layer; didnt addin
            output = Add()([value_output, advantage_output])

            model = tf.keras.Model(inputs=state_input, outputs=output)
            
            model.compile(loss='mse', optimizer=self.optimizer )  ## no metric coz no exact y_pred to compare
            return model
        
    
    def choose_action(self, state):
        reshape_state = np.reshape(state, np.hstack((1, self.feature_shape)))
        
        ##  eps_decay =1 no decay
        self.epsilon *= self.epsilon_decay   
        self.epsilon = max(self.epsilon, self.epsilon_min)   ### no less than epsilon_min
        
        if np.random.uniform() > self.epsilon:
            ## exploit
            # forward feed the observation and get q value for every actions
            actions_value = self.model.predict(state).ravel()
            action = np.argmax(actions_value)
        else:   ## < self.epsilon
            ## explore
            action = np.random.randint(0, self.n_actions)
        
        return action

    
    def train(self, states:np.array, q_targets:np.array):
        ## loss save to the History over the epochs
        history = self.model.fit(states, q_targets, epoch=1, verbose=0, callbacks = [self.tensorboard] if self.tensorboard else None )
        ## {'loss': [1.43580] }  ## epoch=1
        ## {'loss': [1.43580, xxxxx, xxxx] }  ## epoch=3
        q_loss = history.history['loss']
        ### q_loss is cal by tf.reduce_mean(tf.squared_difference(q_target, q_eval)) 
        return q_loss
    
    def save_model(self, filedirname):
        # Calling `save('my_model.h5')` creates a h5 file `my_model.h5`. / .h5 , .tf , .keras / JSON yml 
        # save the architecture of the model + the the weights + the training configuration + the state of the optimizer
        if self.isbuild_net:
            self.model.save(filedirname)
        else:
            return None

    def load_model(self, filedirname):
        # It can be used to reconstruct the model identically.
        self.model = tf.keras.models.load_model(filedirname)
        return self.model




class AgentController(object):
    ## a controller, control single intersection
    def __init__(self, env, feature_shape, n_actions, e_greedy, e_decay, e_min, tensorboard, algo_type= AlgoType.Noraml, build_net=True ):
        self.env = env
        
        self.eval_net = NetModel(feature_shape , n_actions, e_greedy, e_decay, e_min, tensorboard, algo_type, build_net)
        self.target_net = NetModel(feature_shape , n_actions, e_greedy, e_decay, e_min, tensorboard, algo_type, build_net)
        
        self.memory = ExperienceReplay(memory_size= REPLAY_MEM_SIZE)
        
        # total learning step
        self.step_counter = 0
        # replace target net up to steps
        self.replace_target_iter= 300
        
        ### cost amonst the whole training 
        self.cost_his = []
        
    def replace_target_net(self):
        # check to replace target params
        if self.step_counter % self.replace_target_iter == 0:
            ### taraget net is updated later than eval net by every replace_target_iter times
            weights = self.eval_net.model.get_weights()
            self.target_net.model.set_weights(weights)
            
    def train(self, algo_type=AlgoType.Noraml):
        if algo_type== AlgoType.Noraml:
            ### Normal DQN
            ### update target network params; copy from eval net
            self.replace_target_net()
            
            ## get batch of <st,at,rt,st+1> Experience replay transition for training
            states, actions, rewards, next_states, is_done = self.memory.sampling(batch_size= BATCH_SIZE)

            next_q_targets = self.target_net.model.predict(next_states)
            next_q_targets_max = np.max(next_q_targets, axis=1)
            
            q_targets = self.target_net.model.predict(states)
            
            q_targets[range(BATCH_SIZE), actions] = rewards + GAMMA * next_q_targets_max * (1-is_done)  ## if is_done with sim then only reward no next action next state; so *0
            """
            For example in this batch I have 2 samples and 3 actions:
            q_eval =
            [[1, 2, 3],
                [4, 5, 6]]

            q_target = q_eval =
            [[1, 2, 3],
                [4, 5, 6]]

            Then change q_target with the real q_target value w.r.t the q_eval's action.
            For example in:
                sample 0, I took action 0, and the max q_target value is -1;
                sample 1, I took action 2, and the max q_target value is -2:
            q_target =
            [[-1, 2, 3],
                [4, 5, -2]]

            So the (q_target - q_eval) becomes:
            [[(-1)-(1), 0, 0],
                [0, 0, (-2)-(6)]]
                
            ****// ****// ****// ****// ****// ****// ****// ****//
            We then backpropagate this error w.r.t the corresponding action to network,
            leave other action as error=0 cause we didn't choose it.
            ****// ****// ****// ****// ****// ****// ****// ****//
            
            """
            
            ### train eval net
            q_loss = self.eval_net.train(states, q_targets)
            
            return q_loss
        
        elif algo_type== AlgoType.Double:
            ### Double DQN
            ### update target network params; copy from eval net
            self.replace_target_net()
            
            ## get batch of <st,at,rt,st+1> Experience replay transition for training
            states, actions, rewards, next_states, is_done = self.memory.sampling(batch_size= BATCH_SIZE)
            
            ### input next state into eval net , select action from eval net q value with argmax
            maxq_actions = np.argmax(self.eval_net.model.predict(next_states), axis=1)
            ### get above action next_q_values in the target model
            next_q_values = self.target_net.model.predict(next_states)[range(BATCH_SIZE), maxq_actions]
            
            q_targets[range(BATCH_SIZE), actions] = rewards + GAMMA * next_q_values * (1- is_done)  ## if is_done with sim then only reward no next action next state; so *0      
            ### train eval net
            q_loss = self.eval_net.train(states, q_targets)
            
            return q_loss
        

            
    def learn(self, max_ep= MAX_EPISODE, max_timestep = MAX_EP_STEPS, min_Lr_step=900, Lr_iter=5, algo_type=AlgoType.Normal):
        for ep in range(max_ep):
            state = self.env.reset()
            t = 0
            reward_tracker = []
            while True:
                action = self.eval_net.choose_action(state)
                next_state, reward, is_done = self.env.step(action)
                self.memory.store_transition(state, action, reward , next_state, is_done)
            
                reward_tracker.append(reward)
                
                if (t > min_Lr_step) and (t % Lr_iter == 0):
                    ### after sim start min_Lr_step sec afterwards then train by every Lr_iter sec
                    q_loss = self.train(algo_type)
                    
                    if OUTPUT_GRAPH:
                        ### record the loss value
                        self.cost_his.append(q_loss)

                if is_done  or t >= max_timestep:
                    ep_reward_sum = sum(reward_tracker)

                    if 'episode_reward' not in globals():
                        episode_reward = ep_reward_sum
                    else:
                        ### consider previous episode reward in; ratio self defined
                        episode_reward = episode_reward * 0.95 + ep_reward_sum * 0.05
                    
                    print("episode:{episode}  reward:{reward}".format(episode = ep, reward= episode_reward) )
                    wandb.log({'Reward': episode_reward})
                    break
                
                ### swap observation
                state = next_state
                t+=1
                ### used for update the target network
                step_counter+= 1
                
    def get_eval_net_model(self):
        return self.eval_net 
    
    def get_target_net_model(self):
        return self.target_net 
    
    def get_cost_his(self):
        return self.cost_his 