# Implementation Assignment 1

Use python 2.7 
(Only 2.7 had matplotlib in babylon. Should still work in python 3.6 but ymmv)

1) install modules in requirements.txt
2) Ensure that "PA1_train.csv", "PA1_test.csv", and "PA1_dev.csv" are in a data subdirectory
3) You can run the script with `python PA1.py`
  If you want to run specific parts you can do this with the `-p` argument. i.e. running:
  ```
  python PA1.py -p 1 3
  ```
  will run parts 1 and 3.  
4) After running it should output results to the terminal and save plots into the data subdirectory. These plots will be named part_x_discription_of_content.png.
5) if part 2 was run it should also output a predition file for the test data in the root directory named `pred.csv`


Also, the gradient decent and SSE functions are a part of the helper class in `Helper_Class.py` and the preprocessing code is done in the preprocess class in `preprocess.py`
