# Usage


## main
To the input – a txt file (should be placed to the DATA_PATH)    
At the output – a json file (will be saved to the RESULT_PATH)

    usage: main.py [-h] [-vs VS] [-i I] path
    
    positional arguments:
      path        path from DATA_PATH to your txt file
    
    optional arguments:
      -h, --help  show this help message and exit
      -vs VS      size of word-vectors
      -i I        quantity of training loops


## tests
example:

    python3 tests/synonyms.py HP3.json ron -cnt 5
