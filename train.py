import numpy as np
import tensorflow as tf
import json
import argparse
from expmem import ExperienceMemory
from qlearner import QLearner
import game
from pathlib import Path
import matplotlib.pyplot as plt
import datetime
import signal
import sys

#TODO make different parse args for outputting normal images and transition images
#TODO use float or uint8 in states etc, in experience we use int, but in Game we already use float between 0 and 1

class QTrainer:

    def __init__(self,config_or_model,load_model=False):
        self.config = None
        self.model_loaded = False
        if load_model:
            print("Loading model from: {}".format(config_or_model))
            load_path = Path(config_or_model)
            if (not load_path.exists()) or (not load_path.is_dir()):
                print("Error: directory doesn't exist")

            config_filename = load_path.joinpath("config.json")
            self.config = self.load_config(str(config_filename))
        else:
            self.config = self.load_config(config_or_model)
        
        self.game_name = self.config["game"]
        self.game = None
        if self.game_name == "snake":
            self.game =  game.Snake
        elif self.game_name == "box":
            self.game = game.Box
        else:
            print("Error: unknown game {}".format(self.game_name))
        
        self.nn_config = self.config["nn"] 
        self.memory_size = self.config["memory_size"]
        self.memory_alpha = self.config["memory_alpha"]
        self.memory_beta_start = self.config["memory_beta_start"]
        self.memory_beta_end = self.config["memory_beta_end"]
        self.memory_beta_num_steps = self.config["memory_beta_num_steps"]
        self.memory_beta_step = (self.memory_beta_end - self.memory_beta_start)/self.memory_beta_num_steps
        self.exp_memory_start_size = self.config["memory_start_size"]
        self.width = self.config["width"]
        self.height = self.config["height"]
        self.image_scale_factor = self.config["image_scale_factor"]
        self.num_goals = self.config["num_goals"]
        self.img_width = self.width*self.image_scale_factor
        self.img_height = self.height*self.image_scale_factor
        self.num_img_channels = 3
        self.num_actions = self.game.num_actions

        self.epsilon_start = self.config["epsilon_start"]
        self.epsilon_min = self.config["epsilon_min"]
        self.num_epsilon_steps = self.config["num_epsilon_steps"]
        self.epsilon_step = (self.epsilon_start - self.epsilon_min)/self.num_epsilon_steps

        #TODO better mechanism here
        self.scale_reward_max = None
        if "scale_reward_max" in self.config:
            self.scale_reward_max = self.config["scale_reward_max"]
            self.game.max_reward *= self.scale_reward_max
            self.game.min_reward *= self.scale_reward_max
            self.game.empty_reward *= self.scale_reward_max 
            print("Scaling rewards by {}".format(self.scale_reward_max))

        self.max_steps = self.config["max_steps"]
        self.output_freq = self.config["output_freq"]
        self.update_freq = self.config["update_freq"]
        self.target_network_update_mode = self.config["target_network_update_mode"]
        self.target_network_update_tau = None
        self.target_network_update_freq = None
        if self.target_network_update_mode == "hard":
            self.target_network_update_freq = self.config["target_network_update_freq"]
        else:
            self.target_network_update_tau = self.config["target_network_update_tau"] 
        self.eval_freq = self.config["eval_freq"]
        self.eval_steps = self.config["eval_steps"]
        self.tensorboard_log_freq = self.config["tensorboard_log_freq"]
        self.tensorboard_log_path = self.config["tensorboard_log_path"]
        self.save_freq = self.config["save_freq"]
        self.save_path = self.config["save_path"]
        
        self.batch_size = self.config["batch_size"]
        
        self.curr_step = 0
        self.epsilon = self.epsilon_start
        self.memory_beta = self.memory_beta_start
        self.best_average_score = 0

        self.exp_memory = ExperienceMemory(self.memory_size,self.img_width,self.img_height,self.num_img_channels,self.memory_alpha)
        self.qlearner = None
        if load_model:
            load_path = str(Path(config_or_model).joinpath("nn").joinpath("model"))
            self.qlearner = QLearner(self.nn_config,self.num_actions,self.img_width,self.img_height,self.num_img_channels,self.memory_size,load_model=load_path,target_network_update_tau=self.target_network_update_tau)
            self.curr_step = self.config["curr_step"]
            self.epsilon = self.config["epsilon"]
            self.memory_beta = self.config["memory_beta"]
            self.best_average_score = self.config["best_average_score"]
            print("Model loaded successfully")
            self.model_loaded = True
        else:
            self.qlearner = QLearner(self.nn_config,self.num_actions,self.img_width,self.img_height,self.num_img_channels,self.memory_size,target_network_update_tau=self.target_network_update_tau)

        if self.tensorboard_log_freq > 0:
            self.qlearner.add_tensorboard_ops(self.tensorboard_log_path)

    def get_game(self):
        return self.game(self.width,self.height,self.image_scale_factor,self.num_goals)

    def init_random_exp_memory(self,size):
        if size > self.memory_size:
            size = self.memory_size 

        game = self.get_game()
        self.exp_memory.add(game.get_state(),0,0,0)
        for i in range(size):
            random_action = np.random.randint(0,self.num_actions)
            reward, is_terminal = game.execute_action(random_action)    
            state = game.get_state()
            self.exp_memory.add(state,random_action,reward,is_terminal)
            if is_terminal:
                game.reset()
                self.exp_memory.add(game.get_state(),0,0,0)

    def init_exp_memory(self,size):
        if size > self.memory_size:
            size = self.memory_size 

        game = self.get_game()
        self.exp_memory.add(game.get_state(),0,0,0)
        for i in range(size):
            action = 0
            if np.random.rand() < self.epsilon:
                action = np.random.randint(0,self.num_actions)
            else:
                action = self.qlearner.compute_action(game.get_state())[0]  
            reward, is_terminal = game.execute_action(action)    
            state = game.get_state()
            self.exp_memory.add(state,action,reward,is_terminal)
            if is_terminal:
                game.reset()
                self.exp_memory.add(game.get_state(),0,0,0)
       
    def train(self):
        if self.model_loaded:
            self.init_exp_memory(self.exp_memory_start_size)
        else:
            self.init_random_exp_memory(self.exp_memory_start_size)

        total_reward = 0.0
        games_played = 1

        game = self.get_game()
        self.exp_memory.add(game.get_state(),0,0,0)

        while self.curr_step < self.max_steps:
            action = 0
            if np.random.rand() < self.epsilon:
                action = np.random.randint(0,self.num_actions)
            else:
                action = self.qlearner.compute_action(game.get_state())[0]  

            reward, is_terminal = game.execute_action(action)
            self.exp_memory.add(game.get_state(),action,reward,is_terminal)
            if is_terminal:
                game.reset()
                self.exp_memory.add(game.get_state(),0,0,0)
                games_played += 1

            total_reward += self.renormalize_reward(reward)
        
            #compute next epsilon   
            self.epsilon = np.maximum(self.epsilon_min,self.epsilon-self.epsilon_step)
            self.memory_beta = np.minimum(self.memory_beta_end,self.memory_beta+self.memory_beta_step)
            
            if self.curr_step % self.update_freq == 0:
                s,a,r,s2,t,indices,p_values = self.exp_memory.sample(self.batch_size)
                ########
                #TODO DEBUG CODE HERE
                #if self.curr_step < 10000*self.update_freq:
                #    ri = np.random.randint(self.batch_size)
                #    self.save_transition(s[ri],a[ri],r[ri],s2[ri],t[ri],"eval_img",curr_step//self.update_freq)
                write_summary = False
                if (self.tensorboard_log_freq > 0) and (self.curr_step % self.tensorboard_log_freq == 0):
                    write_summary = True
                
                #TODO don't divide beta by 2 here, do it somewhere else
                _, _, td = self.qlearner.train_step(s,a,r,s2,t,p_values,self.memory_beta/2.0,write_summary=write_summary)
                self.exp_memory.update_p(indices,td)

            if self.target_network_update_mode == "soft":
                if self.curr_step % self.update_freq == 0:
                    self.qlearner.update_target_network()
            else:
                if self.curr_step % self.target_network_update_freq == 0:
                    self.qlearner.update_target_network()

            if self.curr_step % self.output_freq == 0:
                average_reward = total_reward / games_played 
                total_reward = 0
                games_played = 1
                print("step: {}  epsilon: {}  average reward per game: {}".format(self.curr_step,self.epsilon,average_reward))


            if (self.curr_step % self.eval_freq == 0):
                score, num_games, average, max_score = self.eval(self.eval_steps)
                print("Evaluating model with {} steps:".format(self.eval_steps))
                print("Total score: {}  Games: {}  Average: {}  Max: {}".format(score,num_games,average,max_score))
                if average >= self.best_average_score:
                    print("Improved average score")
                    print("Saving model...")    
                    self.save()
                    self.best_average_score = average
                #add average score to tensorboard
                summary=tf.Summary()
                summary.value.add(tag='average_score', simple_value=average)
                summary.value.add(tag='max_score', simple_value=max_score)
                self.qlearner.summary_writer.add_summary(summary,self.curr_step)

            #if (self.curr_step > 0) and (self.curr_step % self.save_freq == 0):
            #    score, num_games, average = self.eval(self.eval_steps)
            #    if average >= self.best_average_score:
            #        print("Improved average score")
            #        print("Saving model...")    
            #        self.save()
            #        self.best_average_score = average

            self.curr_step += 1
        
      
        
    def eval(self,num_steps):
        game = self.get_game()
        total_score = 0.0
        current_score = 0.0
        num_games = 1.0
        max_score = 0.0
        for i in range(num_steps):
            action = self.qlearner.compute_action(game.get_state())[0]  
            reward, is_terminal = game.execute_action(action)
            reward = self.renormalize_reward(reward)
            current_score += reward
            total_score += reward
            if is_terminal:
                game.reset()
                if i < (num_steps-1):
                    num_games += 1
                    if current_score > max_score:
                        max_score = current_score
                    current_score = 0
                    

        average = total_score/num_games

        return total_score, num_games, average, max_score

    def renormalize_reward(self,reward):
        if not self.scale_reward_max is None:
            return reward/self.scale_reward_max
        else:
            return reward

    
    def load_config(self,filename):
        result = None
        with open(filename,'r') as fp:
            result = json.load(fp)
        return result

    def save(self):
        base_path = Path(self.save_path)
        if not base_path.exists():
            base_path.mkdir()

        date_str = datetime.datetime.today().strftime("%Y-%m-%d--%H-%M")
        save_path = date_str+"--step"+str(self.curr_step)
        save_path = base_path.joinpath(save_path)

        #create path if it doesn't exist
        if not save_path.exists():
            save_path.mkdir()

        self.config["epsilon"] = self.epsilon
        self.config["curr_step"] = self.curr_step
        self.config["memory_beta"] = self.memory_beta
        self.config["best_average_score"] = self.best_average_score

        #save config
        config_filename = save_path.joinpath("config.json")
        with config_filename.open('w') as fp:
            json.dump(self.config,fp,indent=4)

        #save neural network
        nn_path = save_path.joinpath("nn")
        if not nn_path.exists():
            nn_path.mkdir()

        self.qlearner.save_model(str(nn_path.joinpath("model")))


    ####################
    #ONLY FOR DEBUG#####
    def eval_with_images(self,num_steps,path):
        image_id = 0
        game = self.get_game()
        #image1 = np.copy(sb.get_state())
        self.save_image(game.get_state(),path,image_id,0,0,0,0.0)
        total_score = 0
        games_finished = 0
        max_game_score = 0
        current_game_score = 0.0
        for i in range(num_steps):
            image_id += 1
            #TODO make this always random or no? probably not if we want to test network, maybe make it as extra cli argument
            action = self.qlearner.compute_action(game.get_state())[0]  
            #action = np.random.randint(0,self.num_actions)
            reward, is_terminal = game.execute_action(action)
            reward = self.renormalize_reward(reward)
            #image2 = sb.get_state()          
            #self.save_transition(image1,action,reward,image2,is_terminal,path,image_id)
            total_score += reward
            current_game_score += reward
            self.save_image(game.get_state(),path,image_id,action,reward,is_terminal,current_game_score)
            #image1 = np.copy(sb.get_state())            
            if is_terminal:
                game.reset()
                games_finished += 1
                if current_game_score > max_game_score:
                    max_game_score = current_game_score
                current_game_score = 0.0
                self.save_image(game.get_state(),path,image_id,action,reward,is_terminal,current_game_score)

        print("Max score: {}".format(max_game_score))

    def find_max_games(self,num_steps,path,score_threshold):
        image_id = 0
        game = self.get_game()
        frames = []
        frames.append((np.copy(game.get_state()),0.0))
        max_game_score = 0
        current_game_score = 0.0
        for i in range(num_steps):
            if i % (num_steps//100) == 0:
                print("At step {}".format(i))
            action = self.qlearner.compute_action(game.get_state())[0]  
            reward, is_terminal = game.execute_action(action)
            reward = self.renormalize_reward(reward)
            current_game_score += reward
            frames.append((np.copy(game.get_state()),current_game_score))            
            if is_terminal:
                game.reset()
                if current_game_score > max_game_score:
                    max_game_score = current_game_score

                if current_game_score > score_threshold:
                    for frame in frames:
                        self.save_image(frame[0],path,image_id,0,0,0,frame[1])
                        image_id += 1

                frames = []
                frames.append((np.copy(game.get_state()),0.0))
                current_game_score = 0.0
                
        print("Max score: {}".format(max_game_score))

    def test_experience_memory(self,num_steps,path):
        image_id = 0
        self.init_random_exp_memory(self.exp_memory_start_size)
        s,a,r,s2,t = self.exp_memory.sample(num_steps)
        for i in range(num_steps):
            image_id += 1
            action = a[i]  
            reward = r[i]
            is_terminal = t[i]
            self.save_transition(s[i],action,reward,s2[i],is_terminal,path,image_id)

    def save_transition(self,s,a,r,s2,t,path,image_id):
       self.save_image(self.combine_images(s,s2),path,image_id,a,r,t) 


    def combine_images(self,image1,image2,sep_width=10):
        image1 = np.squeeze(image1)
        image2 = np.squeeze(image2)
        shape = image1.shape
        sep = np.ones([shape[0],sep_width,self.num_img_channels],dtype=float) 
        frames1 = []
        frames2 = []
        for j in range(self.num_frames):
            start_index = j*self.num_img_channels
            end_index = (j+1)*self.num_img_channels
            frames1.append(image1[:,:,start_index:end_index])
            frames2.append(image2[:,:,start_index:end_index])
            if j != (self.num_frames-1):
                frames1.append(sep)
                frames2.append(sep)
        
        image1 = np.concatenate(frames1,axis=1)
        image2 = np.concatenate(frames2,axis=1)
        
        shape = image1.shape
        sep = np.ones([sep_width,shape[1],self.num_img_channels],dtype=float)
                
        return np.concatenate((image2,sep,image1),axis=0)
        
    def save_image(self,img,path,image_id,action,reward,is_terminal,score):
        save_file = Path(path).joinpath("img{}.png".format(image_id))
        with save_file.open('wb') as fp:
            fig = plt.figure()
            plt.imshow(np.squeeze(img),origin="lower")
            plt.axis("off")
            #plt.title("action: {}  reward: {}  terminal: {}".format(self.game.action_names[action],reward,is_terminal))
            plt.title("Score: {}".format(score))
            fig.savefig(fp,bbox_inches='tight',format="png")
            plt.close()

def parse_args():
    parser = argparse.ArgumentParser(description="Train a deep neural network to play games.")
    parser.add_argument("config_or_model",type=str,help="path to configuration file or saved model")
    parser.add_argument("--eval",type=int,default=0,help="play for a given number of steps and output images")
    parser.add_argument("--find_max_games",type=int,default=0,help="play for a given number of steps and output images")
    parser.add_argument("--score_threshold",type=int,default=70,help="only output frames for game with score above this threshold")
    parser.add_argument("--test_exp_mem",type=int,default=0,help="print transition images from experience memory")
    parser.add_argument("--img_path",type=str,default="img",help="path to store images in")
    parser.add_argument("--load_model",action="store_true",help="load a saved model")
    parser.add_argument("--save_on_exit",action="store_true",help="store model when exiting the program")
    args = parser.parse_args()
    return args

def add_exit_handler(qtrainer,save_on_exit):
    def signal_handler(*args):
        if save_on_exit:
            qtrainer.save()
            print("Saved model")

        print("Shutting down...")
        sys.exit(0)
    
    signal.signal(signal.SIGINT,signal_handler)

if __name__ == '__main__':
    args = parse_args()
    qtrainer = QTrainer(args.config_or_model,load_model=args.load_model)
    add_exit_handler(qtrainer,args.save_on_exit)

    if args.eval > 0:
        qtrainer.eval_with_images(args.eval,args.img_path)
    elif args.find_max_games > 0:
        qtrainer.find_max_games(args.find_max_games,args.img_path,args.score_threshold)
    elif args.test_exp_mem > 0:
        qtrainer.test_experience_memory(args.test_exp_mem,args.img_path)
    else:
        qtrainer.train()
