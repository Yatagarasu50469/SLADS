#! /usr/bin/env python3

def computeDifference(array1,array2,type):
    if type == 'D':
        difference=array1!=array2
        difference = difference.astype(float)
    if type == 'C':
        difference=abs(array1-array2)


    return difference
