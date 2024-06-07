import glob
import os
import time
from sage_vrs import vrs_vrpy
problem_paths = glob.glob(os.path.join("Training_Problems", "problem*.txt"))
for prob_path in problem_paths:
    print(os.path.basename(prob_path))
    start = time.time()
    vrs_vrpy(prob_path)
    print(time.time() -start, "seconds")
