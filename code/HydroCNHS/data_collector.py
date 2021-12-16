import logging
logger = logging.getLogger("HydroCNHS.dc")   

class Data_collector(object):
    def __init__(self):
        pass
    def add_field(self, field, data_type={}):
        setattr(self, field, data_type)
    def get_field(self, field): 
        try:
            return getattr(self, field)
        except Exception as e:
            print(e)
            logger.error(e)
    def get_dict(self):
        return self.__dict__    
    