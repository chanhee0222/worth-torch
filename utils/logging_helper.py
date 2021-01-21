import logging
import os


def init_logger(path:str, display=True, suffix=""):
    if not isinstance(suffix, str):
        suffix = str(suffix)
    if len(suffix) > 0:
        suffix = "_" + suffix

    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)
    logger = logging.getLogger()
    logger.handlers = []
    logger.setLevel(logging.DEBUG)
    debug_filename = os.path.join(path, "debug%s.log" %suffix)
    if os.path.isfile(debug_filename):
        for idx in range(9999):
            debug_filename = os.path.join(path, "debug%s.%d.log" %(suffix, idx))
            if not os.path.isfile(debug_filename):
                break

    debug_fh = logging.FileHandler(debug_filename)
    debug_fh.setLevel(logging.DEBUG)

    info_filename = os.path.join(path, "info%s.log" %suffix)
    if os.path.isfile(info_filename):
        for idx in range(9999):
            info_filename = os.path.join(path, "info%s.%d.log" %(suffix, idx))
            if not os.path.isfile(info_filename):
                break

    info_fh = logging.FileHandler(info_filename)
    info_fh.setLevel(logging.INFO)

    if display:
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)

    info_formatter = logging.Formatter('%(asctime)s [%(levelname).1s] %(message)s')
    debug_formatter = logging.Formatter('%(asctime)s [%(levelname).1s] %(message)s | %(lineno)d:%(funcName)s')

    if display:
        ch.setFormatter(info_formatter)

    info_fh.setFormatter(info_formatter)
    debug_fh.setFormatter(debug_formatter)

    if display:
        logger.addHandler(ch)
    logger.addHandler(debug_fh)
    logger.addHandler(info_fh)

    return logger