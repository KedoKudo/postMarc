#!/usr/bin/env python

## 
#  This script is simply used to extract stress information from the Mentat
#  table file. Set Flag to 15 for stress/Y, 14 for strain/x

import sys

fileName = "./"+sys.argv[1]
inFile = open(fileName, "r")

flag = 15
outString = ""

for line in inFile.readlines():
  if "#" in line:
    flag -= 1
    continue
  if flag == 0:
    for item in line.split():
      outString += item + "\n"

inFile.close()

outFile = open("tempStress.txt", "w")
outFile.write(outString)

# print out finish information
print "The stress data is saved in tempStress.txt file, ready for spreadsheet"
