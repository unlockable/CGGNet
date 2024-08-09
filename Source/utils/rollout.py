# -*- coding: utf-8 -*-
# @Author       : William
# @Project      : TextGAN-william
# @FileName     : rollout.py
# @Time         : Created at 2019-03-15
# @Blog         : http://zhiweil.ml/
# @Description  : 
# Copyrights (C) 2018. All Rights Reserved.

import copy
import torch
import torch.nn.functional as F
import config as cfg
import os
import shutil
import hashlib

from utils.text_process import write_tokens_to_files, tensor_to_tokens
from utils.sol_autocompile import parallel_compile
from glob import glob

import hashlib


class ROLLOUT:
    def __init__(self, gen, gpu=True):
        self.gen = gen
        self.old_model = copy.deepcopy(gen)
        self.max_seq_len = gen.max_seq_len
        self.vocab_size = gen.vocab_size
        self.step_size = gen.step_size if gen.name == 'leakgan' else 0
        self.goal_out_size = gen.goal_out_size if gen.name == 'leakgan' else 0
        self.gpu = gpu
        self.code_dic = {}


    def sha256(self, data):
        """Returns the SHA-256 hash of a file."""

        result = hashlib.sha256(data)
        hex_result = result.hexdigest()
        if hex_result in self.code_dic :
            return False
        
        else :
            self.code_dic[hex_result] = True
            return True

    def rollout_mc_search(self, sentences, given_num):
        """
        fill up remain tokens with MC search
        :param sentences: size of batch_size * max_seq_len
        :param given_num:
        :return:
        """
        batch_size = sentences.size(0)

        # get current state
        hidden = self.gen.init_hidden(batch_size)
        # for i in range(given_num):
        inp = sentences[:, :given_num]
        out, hidden = self.gen.forward(inp, hidden, train_step=False, need_hidden=True)
        out = out.view(batch_size, -1, self.vocab_size)[:, -1]

        samples = torch.zeros(batch_size, self.max_seq_len).long()
        samples[:, :given_num] = sentences[:, :given_num]

        if self.gpu:
            samples = samples.cuda()
        # MC search
        for i in range(given_num, self.max_seq_len):
            out = torch.multinomial(torch.exp(out), 1)
            samples[:, i] = out.view(-1).data
            if torch.sum(out) == 0 :
                break
            inp = out.view(-1)            
            out, hidden = self.gen.forward(inp, hidden, train_step=False, need_hidden=True)
            
        self.gen.init_variable(inp.shape[0])
            

        return samples

    def rollout_mc_search_range(self, sentences, given_num, start, end):
        """
        fill up remain tokens with MC search
        :param sentences: size of batch_size * max_seq_len
        :param given_num:
        :return:
        """
        batch_size = sentences.size(0)

        # get current state
        hidden = self.gen.init_hidden(batch_size)
        # for i in range(given_num):
        inp = sentences[:, :given_num]
        inp = sentences[:, start:end]
        
        
        out, hidden = self.gen.forward(inp, hidden, train_step=False, need_hidden=True)
        out = out.view(batch_size, -1, self.vocab_size)[:, -1]

        samples = torch.zeros(batch_size, self.max_seq_len).long()
        samples[:, :given_num] = sentences[:, :given_num]

        if self.gpu:
            samples = samples.cuda()
        # MC search
        for i in range(given_num, self.max_seq_len):
            if i >= start or i <= end :
                continue
            out = torch.multinomial(torch.exp(out), 1)
            samples[:, i] = out.view(-1).data
            if torch.sum(out) == 0 :
                break
            inp = out.view(-1)            
            out, hidden = self.gen.forward(inp, hidden, train_step=False, need_hidden=True)
            
        self.gen.init_variable(inp.shape[0])
            

        return samples
    
    def rollout_mc_search_leakgan(self, sentences, dis, given_num):

        batch_size, seq_len = sentences.size()

        goal_array = torch.zeros((batch_size, seq_len + 1, self.goal_out_size))

        work_hidden = self.gen.init_hidden(batch_size)
        mana_hidden = self.gen.init_hidden(batch_size)
        real_goal = self.gen.goal_init[:batch_size, :]
        out = 0

        if self.gpu:
            goal_array = goal_array.cuda()
            real_goal = real_goal.cuda()

        # get current state
        for i in range(given_num):
            # Get feature.
            dis_inp = torch.zeros(batch_size, seq_len).long()
            dis_inp[:, :i + 1] = sentences[:, :i + 1]  # cut sentences
            leak_inp = sentences[:, i]
            if self.gpu:
                dis_inp = dis_inp.cuda()
                leak_inp = leak_inp.cuda()
            feature = dis.get_feature(dis_inp).unsqueeze(0)

            # Get output of one token
            # cur_goal: batch_size * 1 * goal_out_size
            out, cur_goal, work_hidden, mana_hidden = self.gen(i, leak_inp, work_hidden, mana_hidden,
                                                               feature, real_goal, train=True)

            # Save goal and update last_goal
            goal_array[:, i, :] = cur_goal.squeeze(1)
            if i > 0 and i % self.step_size == 0:
                real_goal = torch.sum(goal_array[:, i - 3:i + 1, :], dim=1)
                if i / self.step_size == 1:
                    real_goal += self.gen.goal_init[:batch_size, :]

        samples = torch.zeros(batch_size, self.max_seq_len).long()
        samples[:, :given_num] = sentences[:, :given_num]

        # MC search
        for i in range(given_num, self.max_seq_len):
            # Sample one token
            out = torch.multinomial(torch.exp(out), 1).view(-1)  # [num_samples] (sampling from each row)
            samples[:, i] = out.data

            # Get feature
            dis_inp = samples
            if self.gpu:
                dis_inp = dis_inp.cuda()
            feature = dis.get_feature(dis_inp).unsqueeze(0)
            leak_inp = out

            # Get output of one token
            # cur_goal: batch_size * 1 * goal_out_size
            out, cur_goal, work_hidden, mana_hidden = self.gen(i, leak_inp, work_hidden, mana_hidden,
                                                               feature, real_goal, train=True)

            # Save goal and update last_goal
            goal_array[:, i, :] = cur_goal.squeeze(1)
            if i > 0 and i % self.step_size == 0:
                real_goal = torch.sum(goal_array[:, i - 3:i + 1, :], dim=1)
                if i / self.step_size == 1:
                    real_goal += self.gen.goal_init[:batch_size, :]

        if self.gpu:
            samples = samples.cuda()

        return samples

    def get_reward(self, sentences, rollout_num, dis, current_k=0):
        """
        get reward via Monte Carlo search
        :param sentences: size of batch_size * max_seq_len
        :param rollout_num:
        :param dis:
        :param current_k: current training gen
        :return: reward: [batch_size]
        """
        with torch.no_grad():
            batch_size = sentences.size(0)
            rewards = torch.zeros([rollout_num * self.max_seq_len, batch_size]).float()
            if self.gpu:
                rewards = rewards.cuda()
            idx = 0
            for i in range(rollout_num):
                for given_num in range(1, self.max_seq_len + 1):
                    samples = self.rollout_mc_search(sentences, given_num)
                    out = dis.forward(samples)
                    out = F.softmax(out, dim=-1)
                    reward = out[:, current_k + 1]
                    rewards[idx] = reward
                    idx += 1
        # rewards = torch.mean(rewards, dim=0)
        rewards = torch.mean(rewards.view(batch_size, self.max_seq_len, rollout_num), dim=-1)

        return rewards

    def fix_comparison_operators(self, code_string):
        code_string = code_string.replace("= >", "=>")
        code_string = code_string.replace("= <", "=<")
        code_string = code_string.replace("< =", "<=")
        code_string = code_string.replace("> =", ">=")
        code_string = code_string.replace("+ =", "+=")
        code_string = code_string.replace("- =", "-=")
        code_string = code_string.replace("* =", "*=")
        code_string = code_string.replace("/ =", "/=")
        code_string = code_string.replace("< <", "<<")
        code_string = code_string.replace("> >", ">>")
        code_string = code_string.replace("! =", "!=")
        code_string = code_string.replace("= =", "==")
        code_string = code_string.replace("= =", "==")
        code_string = code_string.replace("``", "\"")
        code_string = code_string.replace("\'\'", "\"")
        code_string = code_string.replace("& &", "&&")
        code_string = code_string.replace("| |", "||")
        code_string = code_string.replace("* *", "**")      
        
        return code_string

    def sha256_hash_string(self, text):
        # Create a new sha256 hash object
        sha256 = hashlib.sha256()
        
        # Update the hash object with the bytes of the string
        sha256.update(text.encode())
        
        # Get the hexadecimal representation of the digest
        return sha256.hexdigest()
    

    def get_reward_complie_only(self, sentences, rollout_num, dis, compile_path, idx2word_dict, current_k=0):
        """
        get reward via Monte Carlo search
        :param sentences: size of batch_size * max_seq_len
        :param rollout_num:
        :param dis:
        :param current_k: current training gen
        :return: reward: [batch_size]
        """
        with torch.no_grad():
            batch_size = sentences.size(0)
            rewards = torch.zeros([rollout_num * self.max_seq_len, batch_size]).float()
            total_compile_num = 0
            sample_dic = {}
            
            if self.gpu:
                rewards = rewards.cuda()
                
            zero_positions = sentences.eq(0) # (sentences == 0).nonzero(as_tuple=True)[1]
            # Find the index of the sequence with the longest sentence based on EOS position
            first_zero_indices = zero_positions.int().argmax(dim=1)
            max_len = first_zero_indices.argmax().item()
            print("max len : ", max_len)
            for i in range(rollout_num):
                rollout_compile_path = compile_path + "_" + str(i)
                os.mkdir(rollout_compile_path)
                #self.max_seq_len * batch
                for given_num in range(1, max_len + 1):
                    rollout_given_num_compile_path = rollout_compile_path + "/" + str(rollout_num) + "_" + str(given_num)
                    os.mkdir(rollout_given_num_compile_path)
                    
                    samples = self.rollout_mc_search(sentences, given_num)
                    sample_dic[str(i) + str("_") + str(given_num)] = samples
                    
                    save_sample_path = rollout_given_num_compile_path + "/" + 'samples_{}_{:05d}.txt'.format("ADV", given_num)
                    write_tokens_to_files(save_sample_path, tensor_to_tokens(samples, idx2word_dict))
                    
                    #print("comp_result : ", compile_result)

            
            #files = glob("./unique_function_files/*")
            
            print("fix file")
            for i in range(rollout_num):
                rollout_compile_path = compile_path + "_" + str(i)
                #self.max_seq_len * batch
                for given_num in range(1, max_len + 1):
                    rollout_given_num_compile_path = rollout_compile_path + "/" + str(rollout_num) + "_" + str(given_num)
                    
                    files = glob(rollout_given_num_compile_path+"/*.txt")
                    
                    for file in files :
                        f = open(file, "r")
                        content = f.read()
                        f.close()
                        f = open(file, "w")
                        code_string = self.fix_comparison_operators(content)
                        f.write(code_string)
                        f.close()
                
            
            #samples = torch.zeros(batch_size, self.max_seq_len).long()
            
            compilable_tokens = []
            for i in range(rollout_num):
                rollout_compile_path = compile_path + "_" + str(i)
                #self.max_seq_len * batch
                for given_num in range(1, max_len + 1):
                    rollout_given_num_compile_path = rollout_compile_path + "/" + str(rollout_num) + "_" + str(given_num)
                    compile_result = parallel_compile(rollout_given_num_compile_path + "/")                    
                    compiled_num = compile_result.count(1)
                    compilable_indices = [i for i, result in enumerate(compile_result) if result == 1]
                    
                    for idx in compilable_indices :
                        file_path = rollout_given_num_compile_path + "/" + 'samples_{}_{:05d}_{}.txt'.format("ADV", given_num, idx)

                        if self.sha256(sample_dic[str(i) + str("_") + str(given_num)][idx].cpu().numpy().tobytes()) == True :
                            shutil.copy(file_path, "./backup_lstm_default/" + self.sha256_hash_string(file_path) + ".txt")
                            compilable_tokens.append(sample_dic[str(i) + str("_") + str(given_num)][idx])

                    
                    total_compile_num += compiled_num
                    compile_result = torch.tensor(compile_result).cuda()
                    
                    #indices = (tensor == 1).nonzero().squeeze()
                    
                    print("path:",rollout_given_num_compile_path)
                    #compile_result = 1 - compile_result
                    #compile_result = 1 - compile_result
                    print("given_num : ", given_num)
                    print("compile_result : ", compiled_num)
                    print("compiled_num : ", compiled_num)

            print("the number of compilable_tokens:", len(compilable_tokens))
            
            if len(compilable_tokens) == 0 :
                return None
        
        compilable_samples = torch.stack(compilable_tokens, dim=0)
        print("sample shape : ", compilable_samples.shape)

        return compilable_samples


    def get_reward_leakgan(self, sentences, rollout_num, dis, current_k):
        """
        get reward via Monte Carlo search for LeakGAN
        :param sentences: size of batch_size * max_seq_len
        :param rollout_num:
        :param dis:
        :param current_k: current training gen

        :return: reward: batch_size * (max_seq_len / step_size)
        """
        with torch.no_grad():
            batch_size = sentences.size(0)
            rewards = torch.zeros([rollout_num * (self.max_seq_len // self.step_size), batch_size]).float()
            if self.gpu:
                rewards = rewards.cuda()
            idx = 0
            for i in range(rollout_num):
                for t in range(self.max_seq_len // self.step_size):
                    given_num = t * self.step_size + 1  # 1, 5, 9, ..
                    samples = self.rollout_mc_search_leakgan(sentences, dis, given_num)
                    out = dis(samples)
                    out = F.softmax(out, dim=-1)
                    reward = out[:, current_k + 1]
                    rewards[idx] = reward
                    idx += 1

        rewards = rewards.view(batch_size, self.max_seq_len // self.step_size, rollout_num)
        rewards = torch.mean(rewards, dim=-1)
        return rewards

    def get_token_reward(self, sentences, rollout_num, dis, current_k, given_num):
        """
        get reward of each token in sequence via Monte Carlo search
        """
        with torch.no_grad():
            batch_size = sentences.size(0)
            rewards = torch.zeros([rollout_num, batch_size]).float()
            idx = 0
            for i in range(rollout_num):
                samples = self.rollout_mc_search(sentences, given_num)
                out = dis(samples)
                out = F.softmax(out, dim=-1)
                reward = out[:, current_k + 1]
                rewards[idx] = reward
                idx += 1

        rewards = torch.Tensor(rewards).cuda()
        rewards = torch.sum(rewards, dim=0) / rollout_num
        return rewards

    def get_reward_csgan(self, target, rollout_num, csgan_clas):
        pass
