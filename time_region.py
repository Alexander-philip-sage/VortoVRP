import time
import logging
from collections import defaultdict

class TimeRegion():
    def __init__(self):
        self.region_time = defaultdict(lambda: 0)
        self.region_call_count= defaultdict(lambda: 0)
        self.log_summary_called = False
    def log_summary(self):
        self.log_summary_called = True
        print("timing information")
        regions = list(self.region_time.keys())
        all_times = [['region_name', 'ct_calls', 'total_time','time_per_call'] ]
        for reg in regions:
            if self.region_time[reg]>0:
                total_time = self.region_time[reg]
                call_count = self.region_call_count[reg]
                all_times.append([reg, call_count,total_time,total_time/call_count])
                logging.info(reg+": calls {} total time {} time per call {}".format(call_count, 
                    round(total_time,3), 
                    round(total_time/call_count,6)))
    def track_file_wait(self, name: str, time_waited_s:int):
        '''times are expected in s '''
        self.waiting_for_file[name].append(time_waited_s)
    def track_time(self,name: str, time_passed_s: int):
        '''times are expected in s '''
        self.region_time[name] += time_passed_s
        self.region_call_count[name] +=1

def timer_func(func):
    # This function shows the execution time of
    # the function object passed
    def wrap_func(*args, **kwargs):
        t1 = time.time()
        result = func(*args, **kwargs)
        t2 = time.time()
        time_diff = t2-t1
        #print(f'Function {func.__name__!r} executed in {(time_diff):.4f}s')
        time_region.track_time(func.__name__, time_diff)
        return result
    return wrap_func

time_region = TimeRegion()