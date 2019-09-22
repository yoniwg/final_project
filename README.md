# Project Title

PCFG with hybrid CKY algorithm

## Getting Started

These instructions will guide you how to run the project properly.

### Prerequisites

Run:
```
pip install -r requirement.txt
```

## Running the program

In order to execute the application run:
```
python main.py
```

In addition, there are few modes to run as follow:

For running normal CKY algorithm use '--pure-cky' and add '--ignore-unknown' in order to not use any unknown terminal smoothing
```
python main.py --pure-cky [--ignore-unknown]
```
For running the mode in which all of the predictions use Logistic-Regression model use (Very Slow):
```
python main.py --all-lr
```
In order to add percolation when normalizing grammar use:
percolate
```
python main.py --percolate
```
After one training you can cancel the training for one of the model or both.
Use one of the modifiers: 
```
python main.py --no-train       // train nothing
python main.py --train-terms    // train only terminals
python main.py --train-rules    // train only rules
```

You can also control the amount of sentences to train above:
```
python main.py                  // train 3000
python main.py --train-min      // train 1000
python main.py --train-max      // train 5000
```

## Data resources 

All the data resources are located in `data` folder