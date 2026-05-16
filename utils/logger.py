import os
# from torch.utils.tensorboard import SummaryWriter
# import wandb

class Logger:
    def __init__(self, log_dir, use_tensorboard=False, use_wandb=False, config=None):
        self.log_dir = log_dir
        self.use_tensorboard = use_tensorboard
        self.use_wandb = use_wandb
        
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
            
        # if self.use_tensorboard:
        #     self.writer = SummaryWriter(log_dir=log_dir)
            
        # if self.use_wandb:
        #     wandb.init(project="dsunet-lane-segmentation", config=config)

    def log_scalar(self, tag, value, step):
        # if self.use_tensorboard:
        #     self.writer.add_scalar(tag, value, step)
        # if self.use_wandb:
        #     wandb.log({tag: value}, step=step)
        # Simple print fallback
        pass

    def log_images(self, tag, images, step):
        # if self.use_tensorboard:
        #     self.writer.add_images(tag, images, step)
        pass

    def close(self):
        # if self.use_tensorboard:
        #     self.writer.close()
        # if self.use_wandb:
        #     wandb.finish()
        pass
