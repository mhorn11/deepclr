import math
import os
import os.path as osp
from typing import Dict, Optional

from ignite.engine import Events
from ignite.handlers import Timer
from ignite.metrics import Loss, RunningAverage
import torch

from ..data.build import make_data_loader, DataflowDataLoader
from ..config.config import Config
from ..evaluation.evaluator import Evaluator
from ..models.build import BaseModel, build_model, store_models_code
from ..solver.build import make_optimizer, make_scheduler
from ..solver.schedulers import LRScheduler
from ..utils.checkpoint import Checkpointer, load_checkpoint
from ..utils.logging import create_logger, create_summary_writer
from ..utils.metrics import MetricFunction, get_loss_fn, get_metric_fns

from .engines import EngineOutput, create_trainer, create_evaluator, y_from_engine, loss_from_engine
from .loss import LossFn, LossForward


def train(cfg: Config) -> None:
    # create output dir
    output_dir = cfg.output_dir
    if output_dir and not osp.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    # store config
    cfg.write_file(osp.join(cfg.output_dir, 'config.yaml'), invalid=True, internal=True)
    cfg.model.write_file(osp.join(cfg.output_dir, 'model_config.yaml'), invalid=True, internal=True)

    # store models directory
    store_models_code(osp.join(cfg.output_dir, 'models'))

    # model, optimizer and scheduler
    model = build_model(cfg.model)
    optimizer = make_optimizer(cfg, model)
    scheduler = make_scheduler(cfg, optimizer)

    # make zero gradient update to prevent warning (scheduler is called before optimizer)
    if cfg.optimizer.accumulation_steps > 1:
        optimizer.zero_grad()
        optimizer.step()

    # load checkpoint
    epoch = None
    iteration = None

    if cfg.checkpoint is not None:
        checkpoint_data = load_checkpoint(cfg.checkpoint)

        epoch = checkpoint_data['epoch']
        iteration = checkpoint_data['iteration']

        # load states
        model.load_state_dict(checkpoint_data['model_state_dict'])
        optimizer.load_state_dict(checkpoint_data['optimizer_state_dict'])
        if checkpoint_data['scheduler_state_dict'] is not None:
            scheduler.load_state_dict(checkpoint_data['scheduler_state_dict'])

        # move optimizer state to device
        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(cfg.device)

    # data loaders
    train_loader = make_data_loader(cfg, is_train=True)
    val_loader = make_data_loader(cfg, is_train=False)

    # loss and metrics
    loss_fn = get_loss_fn(cfg)
    metric_fns = get_metric_fns(cfg)

    run_trainer(
        cfg,
        model,
        train_loader,
        val_loader,
        optimizer,
        scheduler,
        loss_fn,
        metric_fns,
        epoch,
        iteration
    )


def run_trainer(
        cfg: Config,
        model: BaseModel,
        train_loader: DataflowDataLoader,
        val_loader: DataflowDataLoader,
        optimizer: torch.optim.Optimizer,
        scheduler: LRScheduler,
        loss_fn: MetricFunction,
        metric_fns: Optional[Dict[str, MetricFunction]] = None,
        epoch: Optional[int] = None,
        iteration: Optional[int] = None
) -> None:
    if metric_fns is None:
        metric_fns = dict()

    # load config
    summary_period = cfg.logging.summary_period
    log_period = cfg.logging.log_period
    checkpoint_period = cfg.logging.checkpoint_period
    validation_period = cfg.logging.validation_period

    output_dir = cfg.output_dir
    device = cfg.device
    alpha = cfg.metrics.running_average_alpha

    label_type = cfg.model.label_type

    # epochs
    if cfg.optimizer.max_iterations is not None:
        iteration_epochs = int(math.ceil(cfg.optimizer.max_iterations / len(train_loader)))
        if cfg.optimizer.max_epochs is not None:
            epochs = cfg.optimizer.max_epochs if cfg.optimizer.max_epochs < iteration_epochs else iteration_epochs
        else:
            epochs = iteration_epochs
    else:
        epochs = cfg.optimizer.max_epochs

    # create logger and summary writer
    logger = create_logger(name='train', save_dir=output_dir)
    writer = create_summary_writer(cfg.output_dir)

    # print config
    logger.info("Training with config:\n")
    logger.info(cfg.dump())

    # store config in summary
    logger.info(f"Start training for {epochs} epochs")
    writer.add_text('cfg', cfg.write_str(invalid=True, internal=True).replace('  ', '--').replace('\n', '  \n'))

    # metrics
    val_metrics = {name: Loss(fn, output_transform=y_from_engine) for name, fn in metric_fns.items()}
    val_metrics['loss_fn'] = Loss(loss_fn, output_transform=y_from_engine)

    train_metrics = {name: RunningAverage(LossFn(fn, output_transform=y_from_engine), alpha=alpha)
                     for name, fn in metric_fns.items()}
    train_metrics['loss_fn'] = RunningAverage(LossFn(loss_fn, output_transform=y_from_engine), alpha=alpha)
    train_metrics['loss'] = RunningAverage(LossForward(output_transform=loss_from_engine), alpha=alpha)

    # trainer and evaluator engines
    trainer = create_trainer(model, optimizer, loss_fn, metrics=train_metrics, device=device,
                             accumulation_steps=cfg.optimizer.accumulation_steps)
    evaluator = create_evaluator(model, metrics=val_metrics, device=device)

    # checkpointer
    checkpointer = Checkpointer(output_dir, n_saved=cfg.logging.checkpoint_n_saved, create_dir=True)

    # timer
    timer = Timer(average=True)
    timer.attach(trainer, start=Events.EPOCH_STARTED, resume=Events.ITERATION_STARTED,
                 pause=Events.ITERATION_COMPLETED, step=Events.ITERATION_COMPLETED)

    @trainer.on(Events.STARTED)
    def load_epoch(engine):
        if epoch is not None:
            engine.state.epoch = epoch
        if iteration is not None:
            engine.state.iteration = iteration

    @trainer.on(Events.ITERATION_COMPLETED(every=log_period))
    def log_training_status(engine):
        loss = engine.state.metrics['loss']
        it = (engine.state.iteration - 1) % len(train_loader) + 1
        logger.info(f"Epoch[{engine.state.epoch}] Iteration[{it}/{len(train_loader)}] Loss: {loss}")

    @trainer.on(Events.ITERATION_COMPLETED(every=summary_period))
    def write_training_summary(engine):
        # metrics
        for key, value in engine.state.metrics.items():
            writer.add_scalar(f'train/{key}', value, engine.state.iteration)

        # loss weights
        loss_weights = model.get_loss_weights()
        if loss_weights is not None:
            for key, value in loss_weights.items():
                writer.add_scalar(f'params/{key}', value, engine.state.iteration)

        # learning rate
        lr = min([grp['lr'] for grp in optimizer.state_dict()['param_groups']])
        writer.add_scalar('params/lr', lr, engine.state.iteration)

    @trainer.on(Events.ITERATION_COMPLETED(every=checkpoint_period))
    def write_checkpoint(engine):
        checkpointer.save_checkpoint(engine, model, optimizer, scheduler)

    if cfg.scheduler.on_iteration:
        @trainer.on(Events.ITERATION_COMPLETED)
        def scheduler_step(engine):
            metrics = engine.state.metrics
            scheduler.step(metric=metrics['loss_fn'])
    elif cfg.scheduler.on_epoch:
        @trainer.on(Events.EPOCH_COMPLETED)
        def scheduler_step(engine):
            metrics = engine.state.metrics
            scheduler.step(metric=metrics['loss_fn'])

    @trainer.on(Events.EPOCH_COMPLETED)
    def print_times(engine):
        metrics = engine.state.metrics
        avg_loss = metrics['loss']
        logger.info(f"Training Results - Epoch: {engine.state.epoch} Avg Loss: {avg_loss}")
        tpb = timer.value() * timer.step_count
        speed = cfg.data_loader.batch_size / timer.value()
        logger.info(f"Epoch {engine.state.epoch} done. Time per batch: {tpb:.3f}[s] Speed: {speed:.1f}[samples/s]")
        timer.reset()

    @trainer.on(Events.COMPLETED)
    def training_completed(engine):
        logger.info("Training completed")
        checkpointer.save_special_checkpoint('final', engine, model, optimizer, scheduler)

    @trainer.on(Events.EXCEPTION_RAISED)
    def exception_raised(engine, e):
        if isinstance(e, KeyboardInterrupt):
            logger.info("KeyboardInterrupt. Stopping training.")
            checkpointer.save_special_checkpoint('interrupt', engine, model, optimizer, scheduler)
        else:
            logger.info("{} raised:".format(type(e).__name__))
            logger.info(str(e))
            checkpointer.save_special_checkpoint('exception', engine, model, optimizer, scheduler)
            raise e

    if val_loader is not None:
        eval_export = Evaluator()

        @trainer.on(Events.ITERATION_COMPLETED(every=validation_period) | Events.COMPLETED)
        def run_validation(engine):
            # reset and run evaluation
            eval_export.reset()
            evaluator.run(val_loader, max_epochs=1)

        @evaluator.on(Events.ITERATION_COMPLETED(every=log_period))
        def log_validation_status(engine):
            logger.info(f"Validation Iteration[{engine.state.iteration}/{len(val_loader)}]")

        @evaluator.on(Events.ITERATION_COMPLETED)
        def store_validation_output(engine):
            output: EngineOutput = engine.state.output
            y_pred, y_gt, aux = output['y_pred'], output['y'], output['aux']

            for i in range(y_gt.shape[0]):
                name = aux['d'][i]
                timestamp = aux['t'][i][-1].item()

                transform_gt = label_type.to_matrix(y_gt[i].cpu().numpy())
                transform_pred = label_type.to_matrix(y_pred[i].cpu().numpy())

                eval_export.add_transforms(name, timestamp, transform_pred, transform_gt)

        @evaluator.on(Events.COMPLETED)
        def validation_results(_engine):
            # logging
            metrics = evaluator.state.metrics
            train_it = (trainer.state.iteration - 1) % len(train_loader) + 1
            metrics_loss_fn = metrics['loss_fn']
            logger.info(f"Validation Results - Epoch[{trainer.state.epoch}] Iteration[{train_it}] "
                        f"Avg Loss: {metrics_loss_fn}")

            # store results
            for key, value in metrics.items():
                writer.add_scalar(f'val/{key}', value, trainer.state.iteration)

            # store step metrics
            total_step_errors = eval_export.get_total_step_errors()
            writer.add_scalar('val/step_t_err', total_step_errors.mean.translation.kitti, trainer.state.iteration)
            writer.add_scalar('val/step_r_err', total_step_errors.mean.rotation.kitti, trainer.state.iteration)

            # store sequence images and metrics
            if cfg.data.sequential:
                for name, fig in eval_export.plot_sequences().items():
                    writer.add_figure(f'val/{name}', fig, trainer.state.iteration)

                writer.add_figure('val/kitti_errors', eval_export.plot_total_kitti_errors(), trainer.state.iteration)
                writer.add_figure('val/segment_errors', eval_export.plot_segment_error_bars(), trainer.state.iteration)

                total_segment_errors = eval_export.get_total_segment_errors()
                writer.add_scalar('val/kitti_t_err', total_segment_errors.mean.translation.kitti,
                                  trainer.state.iteration)
                writer.add_scalar('val/kitti_r_err', total_segment_errors.mean.rotation.kitti,
                                  trainer.state.iteration)

            # lr scheduler step
            if cfg.scheduler.on_validation:
                scheduler.step(metric=metrics['loss_fn'])

                # store learning rate
                lr = min([grp['lr'] for grp in optimizer.state_dict()['param_groups']])
                writer.add_scalar('params/lr', lr, trainer.state.iteration)

    optimizer.zero_grad()
    trainer.run(train_loader, max_epochs=epochs)
