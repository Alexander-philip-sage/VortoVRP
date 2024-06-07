# VortoVRP
2023-10-03 Vorto Algorithmic Challenge


## Env

python 3.10.4 was used no other versions of python have been tested

`python -m pip install pandas numpy scipy vrpy networkx, gurobipy, cplex`

a requirements file was created from `python.exe -m pip freeze > requirement.txt`

# Status

The code runs on the training problems provided. It has not been optimized by a smart algorithm like a savings algorithm. It currently uses a naive single for-loop.

## Running Versions

### v1.0
Basic implementation to parse the problem.txt files and create a solution. Not a good solution. 

commit id: a239f07  

tag: v1.0

mean cost: 90138.6429881012

mean run time: 908.5029602050781ms

### v2.0

vrpy. I referenced a [medium article](https://medium.com/@trentleslie/leveraging-the-vehicle-route-problem-with-pickup-and-dropoff-vrppd-for-optimized-beer-delivery-in-392117d69033) for writing this solution. 

**Testing Parameters**

1. throws error: pulp.apis.core.PulpSolverError: PuLP: cannot execute cplex.exe

   `sol = prob.solve(time_limit=25, cspy=False, solver='cplex')`
2. best value 6670 

   `sol = prob.solve(time_limit=25, cspy=False, solver='cbc', pricing_strategy ='Exact')`
3. throws error: gurobipy.GurobiError: Unknown parameter 'options'

   `sol = prob.solve(time_limit=25, cspy=False, solver='gurobi', pricing_strategy ='Exact')`
4. best value 8676

   `sol = prob.solve(time_limit=25, cspy=False, solver='cbc', pricing_strategy ='Exact', dive=True)`
5. best value 6670

   `sol = prob.solve(time_limit=25, cspy=False, solver='cbc', pricing_strategy ='Exact', greedy=True)`
6. best value 1035

    `prob.minimize_global_span = True`

    `sol = prob.solve(time_limit=25, cspy=False, solver='cbc', pricing_strategy ='Exact')`

**Testing Runtime**
using test.py

problem1: 20s
problem10: 74s
problem11: 73s
problem12: 67s
problem13: 91s
problem14: 86s

whereas the for-loop method solves problem13 in 0.35s

using time_region.py

# Notes

* using `inputPath = os.path.join (args.problemDir, inputFile)` on line 160 of evaluateShared.py will allow it to run in a more operating system agnostic way rather than hard coding forward or backward slash.
* the output variable on line 165 shows that load 2 of problem1.txt is assigned to the first driver, but I'm getting an error saying "load 2 was not assigned to a driver". When printing the variables we can see the carriage return is not being stripped  `solutionLoadIDs {'1': True, '2\r': True, '3\r': True, '4': True, '5\r': True, '6': True, '7\r': True, '8': True, '9\r': True, '10\r': True}`  
 `load_ids ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10']`
* There was an error in the parsing of the output. it was stripping `\n` but not `\r`. line.strip() removes both so I fixed it. 

# Conclusion
I was quite surprised that the vrpy package didn't perform better. I did a lot of digging into the documentation, examples on the documentation site, and a medium example usage. I am assuming that the truck can only carry one load from one pickup customer at a time since this is implied in the description, but it isn't explicitly stated. With more time I would try to get the other solvers cplex and gurobi to work, and I might implement a solution with google's or-tools to compare against my other solutions. 