import logging
import torch.distributed as dist


def create_logger(logging_dir):
    """
    Create a logger that writes to a log file and stdout.
    """
    try:
        # Check if distributed environment is initialized
        if dist.is_initialized() and dist.get_rank() == 0:  # real logger
            logging.basicConfig(
                level=logging.INFO,
                format='[\033[34m%(asctime)s\033[0m] %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S',
                handlers=[logging.StreamHandler(), logging.FileHandler(f"{logging_dir}/log.txt")]
            )
            logger = logging.getLogger(__name__)
        else:  # dummy logger (does nothing)
            logger = logging.getLogger(__name__)
            logger.addHandler(logging.NullHandler())
    except (RuntimeError, ValueError):
        # Distributed environment not initialized, create a basic logger
        if logging_dir is not None:
            logging.basicConfig(
                level=logging.INFO,
                format='[\033[34m%(asctime)s\033[0m] %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S',
                handlers=[logging.StreamHandler(), logging.FileHandler(f"{logging_dir}/log.txt")]
            )
        logger = logging.getLogger(__name__)
    
    return logger