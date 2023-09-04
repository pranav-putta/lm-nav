from abc import ABC, abstractmethod

def get_writer(cfg):
    writer_type = cfg.bc.writer

    if writer_type == 'console':
        return ConsoleWriter()
    else:
        raise NotImplementedError()
    

class BaseWriter(ABC):
    
    def __init__(self):
        pass

    @abstractmethod
    def write(self, log_dict):
        pass


class ConsoleWriter(BaseWriter):

    def __init__(self):
        pass

    def write(self, log_dict):
        print(log_dict)
