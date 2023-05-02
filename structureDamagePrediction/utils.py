from datetime import datetime

class StartEndLogger():
    def __init__(self) -> None:
        self.last_msg = ""
    
    def log(self, msg, no_date = False):
        if no_date:
            print("%s"%(msg))
        else:
            print("%s: %s"%(datetime.now(), msg))
    
    def start(self, msg):
        self.last_msg = msg
        self.log("+++ %s"%(msg))
    
    def end(self, msg = None):
        if msg is None:
            msg = self.last_msg
        self.log("--- %s DONE"%(msg))

