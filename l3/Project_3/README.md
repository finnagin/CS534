# Implementation Assignment 3

Use python 2.7 with pandas, numpy and matplotlib
(Should still work in python 3.6 or 3.7 but ymmv)

1) install modules in requirements.txt
2) Ensure that "PA3_train.csv", "PA3_test.csv", and "PA3_val.csv" are unzipped in a data subdirectory
3) You can run the script with `python PA3.py`
  If you want to run specific parts you can do this with the `-p` argument. i.e. running:
  ```
  python PA3.py -p 1 3
  ```
  will run parts 1 and 3. 
  Also you can run without displaying the plots by adding the `--hide` argument:
  ```
  python PA3.py -p 1 3 --hide
  ```
  will run parts 1 and 3 without displaying the plots.
  By default the code uses a preset seed `1123581321` (first 8 numbers of fibinochi) but if you wish to run using a seed generated with time by using the `-r` argument. i.e. running:
  ```
  python PA3 -p 2 3 -r
  ```
  will run parts 2 and 3 with a seed gnerated from the time and should be different each time.
4) After running it should output results to the terminal and save the plots into the root directory.
5) The zip file comes with the prediction file included but upon running part 3 the prediction file will be regenerated and saved again. (should be the same as before)


The decision tree, random forest, and adaboost functions are located in the files `Decision_Tree_Module.py`, `Random_Forest_Module.py`, and `Ada_Boosted_Decision_Tree_Module.py` respectively.