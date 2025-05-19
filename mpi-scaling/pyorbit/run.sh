#!/bin/bash

for ((i = 1 ; i < 10 ; i = i + 1)); 
do 
    echo "mpi n" "$i"; 
    mpirun -n $i python run.py
done
