# Mutagenesis simulator to create fake but realistic sequence-activity data 

Using realistic SOLD matrix, and positing individual and pairwise (or even higher order) effects of mutations at certain (experiment-defined) positions on protein sequence on its activity ("indel"), this package creates fake data of activity ("indel") to test ML models' recovery of said ground-truth effects, and test active learning paradigms to search for better protein sequences. Such realistic simulators are very useful in testing complex algorithms and their performance as a function of data size and underlying model complexity. 

Here "model complexity" is deifined by the number of "parameters" in the ground-truth---
* How many inidividual weights are non-zero?
* How many epistatic effects are non-zero?
* Are there higher order effects?

There is also experimental design---the balance between "exploitation" and "exploration". For example, a large number of positions mutated in the protein leads to higher exploration, too conservative design leads to only "exploitation" of what's roughly already known. 

Model complexity sets how difficult it is to learn the "biology". A robust method assumes high model complexity so that if the biological reality is simpler, the tested algorithms will work even better than expected---because experiments are expensive and simulation is cheap. 


## Description

SOLD matrix is used a common framekwork---deep mutational scan fits in the same framework. Since I am building the simulator only on the individual and pairwise (epistatic) effects of mutations, diffrent length proteins multiple-sequence-aligned can also be captured in the simulation.   

## Getting Started

### Dependencies

* See requirements.txt 

### Installing

* Not yet a package---working on it

* 
### Executing program [[TO DO]] 

* How to run the program
* Step-by-step bullets
```
code blocks for commands
```

## Help

Any advise for common problems or issues.
```
command to run if program contains helper info
```

## Authors

Swagatam Mukhopadhyay 


## Version History

* 0.1
    * Initial Release

## License


## Acknowledgments

