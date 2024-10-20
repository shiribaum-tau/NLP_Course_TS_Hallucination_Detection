# Requirements
- Make sure there is an entities input file called "entities.txt" with names of entities. The file should be in the same directory as the main file. 
- If you change the output directory to be anywhere other the current ./ directory, make sure this directory exists before running the code.  

# Example of running the code
Using GPU with deterministic model: <br>
python main.py --device gpu <br> --deterministic

Using GPU with non-deterministic model: <br>
python main.py --device gpu --temperature 1.0 --top_p 0.9 <br>

# Output file name format
{entity} _ {model} _ {random seed} _ {deterministic} _ {temperature} _ {top_p} _ {generation_len}.pkl

<br>
log file of the run can be found in "generation_log.log"