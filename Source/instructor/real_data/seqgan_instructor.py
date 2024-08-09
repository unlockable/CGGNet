# -*- coding: utf-8 -*-
# @Author       : William
# @Project      : TextGAN-william
# @FileName     : seqgan_instructor.py
# @Time         : Created at 2019-06-05
# @Blog         : http://zhiweil.ml/
# @Description  : 
# Copyrights (C) 2018. All Rights Reserved.

import torch
import torch.optim as optim

import config as cfg
from instructor.real_data.instructor import BasicInstructor
from models.SeqGAN_D import SeqGAN_D
from models.SeqGAN_G import SeqGAN_G
from utils import rollout
from utils.sol_autocompile import parallel_compile
from utils.data_loader import GenDataIter, DisDataIter, CompDisDataIter
import os
from utils.text_process import write_tokens_to_files, tensor_to_tokens

class SeqGANInstructor(BasicInstructor):
    def __init__(self, opt):
        super(SeqGANInstructor, self).__init__(opt)
        num_layers = 3
        # generator, discriminator
        self.gen = SeqGAN_G(cfg.gen_embed_dim, num_layers, cfg.gen_hidden_dim, cfg.vocab_size, cfg.max_seq_len,
                            cfg.padding_idx, self.word2idx_dict, cfg.batch_size, gpu=cfg.CUDA)
        self.dis = SeqGAN_D(cfg.dis_embed_dim, cfg.vocab_size, cfg.padding_idx, gpu=cfg.CUDA)
        self.init_model()

        # Optimizer
        self.gen_opt = optim.Adam(self.gen.parameters(), lr=cfg.gen_lr)
        self.gen_adv_opt = optim.Adam(self.gen.parameters(), lr=0.001)
        self.dis_opt = optim.Adam(self.dis.parameters(), lr=cfg.dis_lr)
        self.samples = None
        self.compile_result = None
        self.adv_dataloader = None

    def _run(self):
        
        vulnerable_file_path = cfg.vul_path
            
        if not cfg.gen_pretrain:
            self.log.info('Starting Generator MLE Training...')
            self.pretrain_generator(cfg.MLE_train_epoch)
            if cfg.if_save and not cfg.if_test:
                torch.save(self.gen.state_dict(), cfg.pretrained_gen_path)
                print('Save pre-trained generator: {}'.format(cfg.pretrained_gen_path))
        
            
        # ===ADVERSARIAL TRAINING===
        self.log.info('Starting Adversarial Training...')

        rollout_func = rollout.ROLLOUT(self.gen, cfg.CUDA)

        for adv_epoch in range(cfg.ADV_train_epoch):
            self.log.info('-----\nADV EPOCH %d\n-----' % adv_epoch)
            self.sig.update()
            if self.sig.adv_sig:
                self.adv_train_generator(cfg.ADV_g_step, adv_epoch, rollout_func)  # Generator
                #self.train_discriminator(cfg.ADV_d_step, cfg.ADV_d_epoch, 'ADV')  # Discriminator
                self.pretrain_generator(1)
                if adv_epoch % cfg.adv_log_step == 0 or adv_epoch == cfg.ADV_train_epoch - 1:
                    torch.save(self.gen.state_dict(), cfg.adv_gen_path)
        
                    
            else:
                self.log.info('>>> Stop by adv_signal! Finishing adversarial training...')
                break


    def _compile(self, phase, epoch):
        """Save model state dict and generator's samples"""
        if phase != 'ADV':
            torch.save(self.gen.state_dict(), cfg.save_model_root + 'gen_{}_{:05d}.pt'.format(phase, epoch))
        save_sample_path = cfg.save_samples_root + 'samples_{}_{:05d}.txt'.format(phase, epoch)
        samples = self.gen.sample(cfg.batch_size, cfg.batch_size)
        write_tokens_to_files(save_sample_path, tensor_to_tokens(samples, self.idx2word_dict))


    def _test(self):
        print('>>> Begin test...')

        self._run()
        pass

    def pretrain_generator(self, epochs):
        """
        Max Likelihood Pre-training for the generator
        """
        for epoch in range(epochs):
            self.sig.update()
            if self.sig.pre_sig:
                
                pre_loss = self.train_gen_epoch(self.gen, self.train_data.loader, self.mle_criterion, self.gen_opt)
                # ===Test===
                if epoch % cfg.pre_log_step == 0 or epoch == epochs - 1:
                    self.log.info(
                        '[MLE-GEN] epoch %d : pre_loss = %.4f' % (epoch, pre_loss))
                    
                    if cfg.if_save and not cfg.if_test:
                        self._save('MLE', epoch)
                    
                    
            else:
                self.log.info('>>> Stop by pre signal, skip to adversarial training...')
                break
                
                
    def adv_train_generator(self, g_step, adv_epoch, rollout_func):

        total_g_loss = 0
        #os.mkdir(cfg.save_samples_root + "epoch_" +  str(adv_epoch))
        print("test:",self.train_data.loader)
        compile_path = cfg.save_samples_root + "epoch_" +  str(adv_epoch)
        for step in range(g_step):
            self.gen.init_variable(cfg.batch_size)
            inp, target = GenDataIter.prepare(self.gen.sample(cfg.batch_size, cfg.batch_size), gpu=cfg.CUDA)

            # ===Train===
            compilable_tokens = rollout_func.get_reward_complie_only(target, cfg.rollout_num, self.dis, compile_path, self.idx2word_dict)
            
            if compilable_tokens == None :
                continue
            #inp, target = GenDataIter.prepare(compilable_tokens, gpu=cfg.CUDA, t=10)
            adv_batch_size = cfg.batch_size if len(compilable_tokens) > cfg.batch_size else len(compilable_tokens)
            if self.adv_dataloader is None :
                #self.adv_dataloader = GenDataIter("./adversarial_code.txt")
                self.adv_dataloader = GenDataIter(compilable_tokens)
            else :
                self.adv_dataloader.append_data(compilable_tokens) 
            

            self.gen.init_variable(adv_batch_size)
            pre_loss = self.train_gen_epoch(self.gen, self.adv_dataloader.loader, self.mle_criterion, self.gen_opt)
            
            if step % cfg.pre_log_step == 0 or step == epochs - 1:
                self.log.info(
                    '[MLE-GEN] epoch %d : pre_loss = %.4f' % (adv_epoch, pre_loss))
                
                if cfg.if_save and not cfg.if_test:
                    self._save('ADV', step)
                
    def train_discriminator(self, d_step, d_epoch, phase='MLE'):
        """
        Training the discriminator on real_data_samples (positive) and generated samples from gen (negative).
        Samples are drawn d_step times, and the discriminator is trained for d_epoch d_epoch.
        """
        # prepare loader for validate
        global d_loss, train_acc
        pos_samples = self.train_data.target
        neg_samples = self.gen.sample(pos_samples.shape[0], cfg.batch_size)
        for step in range(d_step):
            # prepare loader for training
            
            dis_data = DisDataIter(pos_samples, neg_samples)

            for epoch in range(d_epoch):
                # ===Train===
                d_loss, train_acc = self.train_dis_epoch(self.dis, dis_data.loader, self.dis_criterion,
                                                         self.dis_opt)

            # ===Test===
            self.log.info('[%s-DIS] d_step %d: d_loss = %.4f, train_acc = %.4f,' % (
                phase, step, d_loss, train_acc))

            if cfg.if_save and not cfg.if_test:
                torch.save(self.dis.state_dict(), cfg.pretrained_dis_path)
