import datetime

def get_current_time():
    currentDT = datetime.datetime.now()
    return currentDT.strftime("%Y%m%d_%H%M%S")