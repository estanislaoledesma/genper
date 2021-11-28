#!/usr/bin/env python
# -*- coding: utf-8 -*-
import torch


class CheckpointManager:

    @staticmethod
    def save_checkpoint(model, optimizer, save_path, epoch, min_valid_loss, training_errors, validation_errors, time_elapsed):
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'epoch': epoch,
            'min_valid_loss': min_valid_loss,
            'training_errors': training_errors,
            'validation_errors': validation_errors,
            'time_elapsed': time_elapsed
        }, save_path)

    @staticmethod
    def load_checkpoint(model, load_path, device, optimizer=None):
        checkpoint = torch.load(load_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        if optimizer is not None:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        min_valid_loss = checkpoint['min_valid_loss']
        training_errors = checkpoint['training_errors']
        validation_errors = checkpoint['validation_errors']
        time_elapsed = checkpoint['time_elapsed']

        return model, optimizer, epoch, min_valid_loss, training_errors, validation_errors, time_elapsed

    @staticmethod
    def update_epoch(checkpoint_path, epoch, training_errors, validation_errors, time_elapsed):
        checkpoint = torch.load(checkpoint_path)
        checkpoint['epoch'] = epoch
        checkpoint['training_errors'] = training_errors
        checkpoint['validation_errors'] = validation_errors
        checkpoint['time_elapsed'] = time_elapsed
        torch.save(checkpoint, checkpoint_path)

    @staticmethod
    def save_datasets(train_loader, val_loader, test_loader, save_path):
        torch.save({
            'train_loader': train_loader,
            'val_loader': val_loader,
            'test_loader': test_loader,
        }, save_path)

    @staticmethod
    def load_datasets(load_path, device):
        datasets = torch.load(load_path, map_location=device)
        train_loader = datasets['train_loader']
        val_loader = datasets['val_loader']
        test_loader = datasets['test_loader']

        return train_loader, val_loader, test_loader
