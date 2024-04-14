import numpy as np
import pandas as pd

#Class to analyze different elements and their count
class enumeratedElement:
    elements = np.full(1, np.nan) #Array that stores elements. Cannot have repeating elements
    elementCount = np.zeros(1) #Array stores how many times has each element repeated while attempting to add.
    totalElements = 0

    def __init__(self, size : int):
        self.elements = np.full(size, np.nan)
        self.elementCount = np.zeros(size)
        
    #Adds element to "elements" if it doesn't already exists on the first NaN index. If it does exist, increment elementCount by one.
    def add(self, element):
        
        contains = 0
        for i in range(len(self.elements)):
            if self.elements[i] == element:
                contains = 1
                self.totalElements += 1
                self.elementCount[i] += 1
        
        if contains == 0:
            for i in range(len(self.elements)):
                if np.isnan(self.elements[i]):
                    self.elements[i] = element
                    self.totalElements += 1
                    self.elementCount[i] += 1
                    break
    
    #Returns number of times an element has been included
    def count(self, element) -> int:
        for i in range(len(self.elements)):
            if self.elements[i] == element:
                return self.elementCount[i]
        return 0        
    
    #Removes an x amount of an y element.
    def remove(self, element, amount : int):
        for i in range(len(self.elements)):
            if self.elements[i] == element and (self.elementCount[i] - amount) > 0:
                self.elementCount[i] -= amount
                return 0
            elif self.elements[i] == element and (self.elementCount[i] - amount) <= 0:
                self.elementCount[i] = 0
                self.elements[i] = np.NaN
                return 0
            else:
                print("None of", element, " found.")
                return -1
    
    #Calculates the percentage of an element over the total number of elements.
    def percentageOf(self, element) -> float:       
        for i in range(len(self.elements)):
            if self.elements[i] == element:
                percentage = (self.elementCount / self.totalElements) * 100
                print("Percentage of ", element," is ", percentage)
                return percentage
        return 0
    
    def toDataFrame(self) -> pd.DataFrame:
        data = pd.DataFrame()
        for i in range(len(self.elements)):
            data.add({f"Interval to: {self.elements[i]}": self.elementCount[i]})
        return data

    def dumpInfo(self):
        for i in range(len(self.elements)):
            print("Element ", self.elements[i], "of index", i ,"has count of: ", self.elementCount[i])
        
            
            
    