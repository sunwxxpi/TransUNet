import os
import sys
import random
import logging
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.modules.loss import CrossEntropyLoss
from torch.utils.data import DataLoader
from torchvision import transforms
from tensorboardX import SummaryWriter
from tqdm import tqdm
from utils import PolyLRScheduler, DiceLoss
from datasets.dataset_synapse import Synapse_dataset, RandomGenerator
from datasets.dataset_coca import COCA_dataset, RandomGenerator

def trainer_synapse(args, model, snapshot_path):
    logging.basicConfig(filename=snapshot_path + "/log.txt", level=logging.INFO, format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    
    base_lr = args.base_lr
    num_classes = args.num_classes
    batch_size = args.batch_size * args.n_gpu
    
    db_train = Synapse_dataset(base_dir=args.root_path, 
                               list_dir=args.list_dir, 
                               split="train",
                               transform=transforms.Compose([RandomGenerator(output_size=[args.img_size, args.img_size])]))
    print("The length of train set is: {}".format(len(db_train)))

    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    trainloader = DataLoader(db_train, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True, worker_init_fn=worker_init_fn)
    
    if args.n_gpu > 1:
        model = nn.DataParallel(model)
    model.train()
    
    dice_loss = DiceLoss(num_classes)
    ce_loss = CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=0.99, weight_decay=3e-5)
    
    max_iterations = args.max_epochs * len(trainloader)
    scheduler = PolyLRScheduler(optimizer, initial_lr=base_lr, max_steps=max_iterations)
    
    writer = SummaryWriter(snapshot_path + '/log')
    logging.info("{} iterations per epoch. {} max iterations ".format(len(trainloader), max_iterations))

    iter_num = 0
    max_epoch = args.max_epochs
    
    iterator = tqdm(range(max_epoch), ncols=70)
    for epoch_num in iterator:
        for i_batch, sampled_batch in enumerate(trainloader):
            image_batch, label_batch = sampled_batch['image'], sampled_batch['label']
            image_batch, label_batch = image_batch.cuda(), label_batch.cuda()
            
            outputs = model(image_batch)
            loss_dice = dice_loss(outputs, label_batch, softmax=True)
            loss_ce = ce_loss(outputs, label_batch[:].long())
            loss = loss_ce + loss_dice

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Step scheduler
            scheduler.step()
            current_lr = scheduler.optimizer.param_groups[0]['lr']
            
            iter_num += 1
            writer.add_scalar('info/lr', current_lr, iter_num)
            writer.add_scalar('info/loss_dice', loss_dice, iter_num)
            writer.add_scalar('info/loss_ce', loss_ce, iter_num)
            writer.add_scalar('info/loss_total', loss, iter_num)

            logging.info('iteration %d - loss_dice : %f, loss_ce: %f, loss_total: %f' % (iter_num, loss_dice.item(), loss_ce.item(), loss.item()))

            if iter_num % 20 == 0:
                image = image_batch[1, 0:1, :, :]
                image = (image - image.min()) / (image.max() - image.min())
                
                writer.add_image('train/Image', image, iter_num)
                
                outputs = torch.argmax(torch.softmax(outputs, dim=1), dim=1, keepdim=True)
                writer.add_image('train/Prediction', outputs[1, ...] * 50, iter_num)

                labs = label_batch[1, ...].unsqueeze(0) * 50
                writer.add_image('train/GroundTruth', labs, iter_num)

        save_interval = 25
        if epoch_num > int(max_epoch / 5) and (epoch_num + 1) % save_interval == 0:
            save_mode_path = os.path.join(snapshot_path, 'epoch_' + str(epoch_num) + '.pth')
            torch.save(model.state_dict(), save_mode_path)
            logging.info("save model to {}".format(save_mode_path))

        if epoch_num >= max_epoch - 1:
            save_mode_path = os.path.join(snapshot_path, 'epoch_' + str(epoch_num) + '.pth')
            torch.save(model.state_dict(), save_mode_path)
            logging.info("save model to {}".format(save_mode_path))
            
            iterator.close()
            break

    writer.close()
    return "Training Finished!"

def trainer_coca(args, model, snapshot_path):
    logging.basicConfig(filename=snapshot_path + "/log.txt", level=logging.INFO, format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    
    base_lr = args.base_lr
    num_classes = args.num_classes
    batch_size = args.batch_size * args.n_gpu
    
    db_train = COCA_dataset(base_dir=args.root_path, 
                               list_dir=args.list_dir, 
                               split="train",
                               transform=transforms.Compose([RandomGenerator(output_size=[args.img_size, args.img_size])]))
    print("The length of train set is: {}".format(len(db_train)))

    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    trainloader = DataLoader(db_train, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True, worker_init_fn=worker_init_fn)
    
    if args.n_gpu > 1:
        model = nn.DataParallel(model)
    model.train()
    
    dice_loss = DiceLoss(num_classes)
    ce_loss = CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=0.99, weight_decay=3e-5)
    
    max_iterations = args.max_epochs * len(trainloader)
    scheduler = PolyLRScheduler(optimizer, initial_lr=base_lr, max_steps=max_iterations)
    
    writer = SummaryWriter(snapshot_path + '/log')
    logging.info("{} iterations per epoch. {} max iterations ".format(len(trainloader), max_iterations))

    iter_num = 0
    max_epoch = args.max_epochs
    
    iterator = tqdm(range(max_epoch), ncols=70)
    for epoch_num in iterator:
        for i_batch, sampled_batch in enumerate(trainloader):
            image_batch, label_batch = sampled_batch['image'], sampled_batch['label']
            image_batch, label_batch = image_batch.cuda(), label_batch.cuda()
            
            outputs = model(image_batch)
            loss_dice = dice_loss(outputs, label_batch, softmax=True)
            loss_ce = ce_loss(outputs, label_batch[:].long())
            loss = loss_ce + loss_dice

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Step scheduler
            scheduler.step()
            current_lr = scheduler.optimizer.param_groups[0]['lr']
            
            iter_num += 1
            writer.add_scalar('info/lr', current_lr, iter_num)
            writer.add_scalar('info/loss_dice', loss_dice, iter_num)
            writer.add_scalar('info/loss_ce', loss_ce, iter_num)
            writer.add_scalar('info/loss_total', loss, iter_num)

            logging.info('iteration %d - loss_dice : %f, loss_ce: %f, loss_total: %f' % (iter_num, loss_dice.item(), loss_ce.item(), loss.item()))

            if iter_num % 20 == 0:
                image = image_batch[1, 0:1, :, :]
                image = (image - image.min()) / (image.max() - image.min())
                
                writer.add_image('train/Image', image, iter_num)
                
                outputs = torch.argmax(torch.softmax(outputs, dim=1), dim=1, keepdim=True)
                writer.add_image('train/Prediction', outputs[1, ...] * 50, iter_num)

                labs = label_batch[1, ...].unsqueeze(0) * 50
                writer.add_image('train/GroundTruth', labs, iter_num)

        save_interval = 25
        if epoch_num > int(max_epoch / 5) and (epoch_num + 1) % save_interval == 0:
            save_mode_path = os.path.join(snapshot_path, 'epoch_' + str(epoch_num) + '.pth')
            torch.save(model.state_dict(), save_mode_path)
            logging.info("save model to {}".format(save_mode_path))

        if epoch_num >= max_epoch - 1:
            save_mode_path = os.path.join(snapshot_path, 'epoch_' + str(epoch_num) + '.pth')
            torch.save(model.state_dict(), save_mode_path)
            logging.info("save model to {}".format(save_mode_path))
            
            iterator.close()
            break

    writer.close()
    return "Training Finished!"