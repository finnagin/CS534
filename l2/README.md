# Implementation Assignment 2

Use python 2.7 with pandas, nupy and matplotlib
(Should still work in python 3.6 but ymmv)

1) install modules in requirements.txt
2) Ensure that "PA2_train.csv.zip", "PA2_test.csv.zip", and "PA2_dev.csv.zip" are in a data subdirectory
3) You can run the script with `python PA2.py`
  If you want to run specific parts you can do this with the `-p` argument. i.e. running:
  ```
  python PA1.py -p 1 3
  ```
  will run parts 1 and 3. 
  Also you can run without displaying the plots by adding the `--hide` argument:
  ```
  python PA1.py -p 1 3 --hide
  ```
  will run parts 1 and 3 without displaying the plots.
4) After running it should output results to the terminal and display the plots to screen.
5) After running the prediction files `oplabel.csv`, `aplabel.csv`, and `kplabel.csv` will be in the root directory if parts 1, 2, or 3 were run respectively


The data loading fuctions and the perceptron algorithms are located in the `Perceptron_Algorithms.py` file.