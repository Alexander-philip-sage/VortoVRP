# VortoVRP
2023-10-03 Vorto Algorithmic Challenge

python 3.10.4 was used no other versions of python have been validated

a requirements file was created from `python.exe -m pip freeze > requirement.txt`

## Status

The code runs on the training problems provided. It has not been optimized by a smart algorithm like a savings algorithm. It currently uses a naive single for-loop.

mean cost: 90138.6429881012
mean run time: 908.5029602050781ms

## Notes

* using `inputPath = os.path.join (args.problemDir, inputFile)` on line 160 of evaluateShared.py will allow it to run in a more operating system agnostic way rather than hard coding forward or backward slash.
* the output variable on line 165 shows that load 2 of problem1.txt is assigned to the first driver, but I'm getting an error saying "load 2 was not assigned to a driver". When printing the variables we can see the carriage return is not being stripped  `solutionLoadIDs {'1': True, '2\r': True, '3\r': True, '4': True, '5\r': True, '6': True, '7\r': True, '8': True, '9\r': True, '10\r': True}`  
 `load_ids ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10']`
* There was an error in the parsing of the output. it was stripping `\n` but not `\r`. line.strip() removes both so I fixed it. 