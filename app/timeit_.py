import sys
import time

from functools import wraps


def format_time(total_time: float):
    """
    Format input time to format: '{hours}h {minutes}m {seconds}s'
    :param total_time: time in seconds
    :return: str
    """
    h = total_time // 3600
    min_ = (total_time - h * 3600) // 60
    sec = total_time - h * 3600 - min_ * 60
    txt = '{hours}{minutes}{seconds:.2f}sec'.format(
        hours   = '{:.0f}h '.format(h) if h else '',
        minutes = '{:.0f}min '.format(min_) if min_ else '',
        seconds = sec,
    )
    return txt


def timeit(msg: str = None, output=sys.stdout):
    """
    Decorator. Print time of function`s execution with the message: msg
    :param msg: message to print with the time
    :param output: a file-like object (stream); defaults to the current sys.stdout.
    """
    def _timeit(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start = time.time()
            res = func(*args, **kwargs)
            print('{msg}:\t{time}'.format(msg=msg, time=format_time(time.time() - start)), file=output)
            return res

        nonlocal msg
        if msg is None:
            msg = 'Time to execute {func_name}'.format(func_name=func.__name__)
        return wrapper

    return _timeit


class timeit_context:
    """
    Context manager. Print time of block's execution with message: msg
    :param msg: message to print with the time
    :param output: a file-like object (stream); defaults to the current sys.stdout.
    :return:
    """

    def __init__(self, msg: str = None, output=sys.stdout):
        if msg:
            self.message = msg
        else:
            self.message = 'Time to execute block'
        self.output = output
        self.start_time = time.time()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.output:
            print('{msg}:\t{time}'.format(
                msg=self.message, time=format_time(time.time() - self.start_time)), file=self.output,
            )
