from os.path import join

class TensorBoardLogger:
    def __init__(self, log_dir):
        from tensorboardX import SummaryWriter

        self._summary_writer = SummaryWriter(
            logdir=join(log_dir, "tensorboard"), max_queue=1, flush_secs=1
        )
        self._step = 0

    def write(self, name, value, step=None):
        self._summary_writer.add_scalar(
                tag=name,
                scalar_value=value,
                global_step= step if step else self._step,
            )
        
        if step is None:
            self._step += 1