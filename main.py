import config as C


def main(local_rank):
    import os
    import sys
    import random
    from pathlib import Path
    import numpy as np
    import matplotlib
    import matplotlib.pyplot as plt
    import torch
    from torch import nn
    import torch.nn.functional as F
    import torch.distributed as dist
    import clip
    from utils.pytorch import save, load_model, load_training_state, count_trainable_params, get_system_info, AutocastWrapper
    from utils.optimizer import update_learning_rate
    from utils.training import History, process_history, Timer
    from utils.loss import RMILoss
    from utils import display
    from text_vqa_x import TextVQAX
    from data import create_data_loaders
    from model import SingleScaleModel

    torch.backends.cudnn.benchmark = True

    node_id = int(sys.argv[1])
    device_id = C.devices[local_rank]
    rank = (node_id * len(C.devices)) + local_rank
    torch.cuda.set_device(device_id)
    world_size = len(C.devices) * C.n_nodes

    # seed = rank
    # random.seed(seed)
    # np.random.seed(seed)
    # torch.manual_seed(seed)

    matplotlib.use('Agg')
    storage_path = Path(__file__).parent.absolute()

    dataset = TextVQAX()
    n_classes = dataset.n_classes
    ignore_index = n_classes
    dataset.ignore_index = ignore_index
    dataset.class_colors.append([0, 0, 0])

    os.environ['MASTER_ADDR'] = C.master_address
    os.environ['MASTER_PORT'] = C.master_port
    os.environ['WORLD_SIZE'] = str(world_size)
    os.environ['RANK'] = str(rank)

    dist.init_process_group(backend='nccl', rank=rank, world_size=world_size)

    training_loader, training_sampler, validation_loader = create_data_loaders(dataset)

    model, preprocess = clip.load('ViT-B/32', device)
    model = SingleScaleModel(
        n_channels=384,
        clip_model=model,
        image_encode_dim=384,
        text_encode_dim=384,
        dropout=0.3
    )

    model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
    load_model(storage_path, model, remove_prefix='module.model.')
    model = AutocastWrapper(model).cuda()
    model = nn.parallel.DistributedDataParallel(model, device_ids=[device_id])
    sentence_encoder = sentence_encoder.cuda()
    display.print_now('Connected.')

    criterion = nn.CrossEntropyLoss(ignore_index=ignore_index)
    rmi = RMILoss(n_classes+1, ignore_index=ignore_index, stride=1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=C.learning_rate, weight_decay=C.weight_decay)

    timer = Timer()
    temp_training_history = History()
    temp_validation_history = History()
    scaler = torch.cuda.amp.GradScaler()
    start_epoch, training_history, validation_history, m_iou_history = load_training_state(storage_path, optimizer, scaler, 0, History('training accuracy'), History('validation accuracy'), History('mIoU'))

    epoch_iterations = len(training_loader)
    total_iterations = C.n_epochs * epoch_iterations

    def train_one_epoch():
        training_sampler.set_epoch(epoch)
        model.train()

        for iteration, data in enumerate(training_loader):
            current_iteration = (epoch - 1) * epoch_iterations + iteration
            update_learning_rate(1e-8, 0.0000001, C.learning_rate, current_iteration, total_iterations, optimizer, 'poly', warmup=epoch_iterations * C.warmup, exp_factor=0.9)

            images = data[0].cuda()
            labels = data[1].cuda()
            texts = data[2]

            texts = sentence_encoder.encode(texts, convert_to_tensor=True)

            with torch.cuda.amp.autocast():
                logits = model(images, texts)
                loss = rmi(logits, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

            loss = loss.item()
            predictions = logits.detach().argmax(1)
            valid = labels.ne(ignore_index)
            TP = (predictions.eq(labels) & valid).sum().item()
            accuracy = TP / (valid.sum().item() or 0.01)

            temp_training_history.add(loss, accuracy)

        with torch.no_grad():
            images = images.cpu().numpy().astype(np.uint8)
        labels = labels.cpu().numpy()
        predictions = predictions.cpu().numpy()
        display.images_segmentations_predictions(images[:4], labels[:4], predictions, dataset.class_colors, ignore_index, ignore_index, resolution=4, save_path=storage_path / f'training_{rank}.png')

    def validate():
        confusion_matrix = torch.zeros(n_classes, n_classes, dtype=torch.int64, device='cuda')
        model.eval()
        with torch.no_grad():
            for images, labels, texts in validation_loader:
                images = images.cuda()
                labels = labels.cuda()

                texts = sentence_encoder.encode(texts, convert_to_tensor=True)

                with torch.cuda.amp.autocast():
                    if len(C.validation_scales) == 1:
                        logits = model(images, texts)
                    else:
                        image_size = images.shape[2:]
                        logits = None
                        for scale in C.validation_scales:
                            temp_images = F.interpolate(images, scale_factor=scale, mode='bicubic', align_corners=False, recompute_scale_factor=False)
                            temp_logits = model(temp_images, texts)
                            temp_logits = F.interpolate(temp_logits, size=image_size, mode='bilinear', align_corners=False)
                            logits = temp_logits if logits is None else logits + temp_logits

                            temp_images = torch.flip(temp_images, [-1])
                            temp_logits = model(temp_images, texts)
                            temp_logits = F.interpolate(temp_logits, size=image_size, mode='bilinear', align_corners=False)
                            logits += torch.flip(temp_logits, [-1])

                    loss = criterion(logits, labels)

                loss = loss.item()
                predictions = logits.argmax(1)
                valid = labels.ne(ignore_index)
                TP = (predictions.eq(labels) & valid).sum().item()
                accuracy = TP / (valid.sum().item() or 0.01)

                temp_validation_history.add(loss, accuracy)

                indices = n_classes * predictions[valid].flatten() + labels[valid].flatten()
                confusion_matrix += torch.bincount(indices, minlength=n_classes ** 2).reshape(n_classes, n_classes)

            images = images.cpu().numpy().astype(np.uint8)
            labels = labels.cpu().numpy()
            predictions = predictions.cpu().numpy()
            display.images_segmentations_predictions(images, labels, predictions, dataset.class_colors, ignore_index, ignore_index, resolution=5, save_path=storage_path / f'validation_{rank}.png')

        dist.all_reduce(confusion_matrix, op=dist.ReduceOp.SUM)
        m_ious = confusion_matrix.diag() / (confusion_matrix.sum(dim=1) + confusion_matrix.sum(dim=0) - confusion_matrix.diag() + 1e-15)
        m_iou_history.add(None, m_ious[1].item(), epoch)

    title_1 = count_trainable_params(model)

    figure = plt.figure(figsize=(18, 8))
    figure.subplots_adjust(top=0.80)
    loss_axes = figure.add_subplot(1, 2, 1)
    accuracy_axes = figure.add_subplot(1, 2, 2)

    for epoch in range(start_epoch + 1, C.n_epochs + 1):
        train_one_epoch()
        process_history(temp_training_history, training_history, epoch)

        if epoch % C.validation_epoch == 0:
            validate()
            process_history(temp_validation_history, validation_history, epoch)

        if rank == 0:
            loss_axes.clear()
            accuracy_axes.clear()

            title_2 = training_history.display()
            loss_axes.plot(training_history.iterations, training_history.losses, color='#0092cc', label='Training')
            accuracy_axes.plot(training_history.iterations, training_history.accuracies, color='#0092cc', label='Training')
            accuracy_axes.plot(*training_history.max_accuracies(), color='#0092cc', marker='o')

            title_3 = validation_history.display()
            loss_axes.plot(validation_history.iterations, validation_history.losses, color='#dcd427', label='Validation')
            accuracy_axes.plot(validation_history.iterations, validation_history.accuracies, color='#dcd427', label='Validation')
            accuracy_axes.plot(*validation_history.max_accuracies(), color='#dcd427', marker='o')

            title_4 = m_iou_history.display()
            accuracy_axes.plot(m_iou_history.iterations, m_iou_history.accuracies, color='#ff3333', label='mIoU')
            accuracy_axes.plot(*m_iou_history.max_accuracies(), color='#ff3333', marker='o')

            title_5 = timer.one_epoch(C.n_epochs, epoch)
            title_6 = get_system_info()

            title = f'{title_1}\n{title_2}\n{title_3}\n{title_4}\n{title_5}\n{title_6}'
            figure.suptitle(title)

            loss_axes.set_ylabel('Loss')
            loss_axes.legend()
            accuracy_axes.set_ylabel('Accuracy')
            accuracy_axes.legend()
            figure.savefig(storage_path / 'learning_curve.png', bbox_inches='tight')

            if epoch % C.save_epoch == 0:
                save(storage_path, model, optimizer, scaler, epoch, training_history, validation_history, m_iou_history)


if __name__ == '__main__':
    import torch
    torch.multiprocessing.spawn(main, nprocs=len(C.devices))
