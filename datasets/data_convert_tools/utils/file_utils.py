from loguru import logger

def check_dir(p, print_info=False):
    if p.is_dir():
        if print_info:
            logger.info(f"already exists {p}")
    else:
        p.mkdir(parents=True)
        if print_info:
            logger.info(f"create dir-{p}")