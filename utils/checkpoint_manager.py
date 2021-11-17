#!/usr/bin/env python
# -*- coding: utf-8 -*-
import torch


class CheckpointManager:

    @staticmethod
    def save_checkpoint(model, optimizer, save_path, epoch, min_valid_loss):
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'epoch': epoch,
            'min_valid_loss': min_valid_loss
        }, save_path)

    @staticmethod
    def load_checkpoint(model, optimizer, load_path):
        checkpoint = torch.load(load_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        min_valid_loss = checkpoint['min_valid_loss']

        return model, optimizer, epoch + 1, min_valid_loss
