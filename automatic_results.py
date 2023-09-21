#This script aims to predict all the damage types of all the layers at once

import os

#argument input combinations for all the damage type of all the layers
combinations =[('-t 2', '-ro 1', '-co 0'),
               ('-t 2', '-ro 3', '-co 0'),
               ('-t 2', '-ro 0', '-co 0'),
               ('-t 2', '-ro 0', '-co 1'),
               ('-t 2', '-ro 0', '-co 2'),
               ('-t 2', '-ro 0', '-co 3'),
               ('-t 2', '-ro 2', '-co 0'),
               ('-t 2', '-ro 2', '-co 1'),
               ('-t 2', '-ro 2', '-co 2'),
               ('-t 2', '-ro 2', '-co 3'),
               ('-t 2', '-ro 4', '-co 0'),
               ('-t 2', '-ro 4', '-co 1'),
               ('-t 2', '-ro 4', '-co 2'),
               ('-t 2', '-ro 4', '-co 3')]


for tupleindex, row, col in combinations:
    command = f'python run_experiment.py {tupleindex} {row} {col}'
    os.system(command)