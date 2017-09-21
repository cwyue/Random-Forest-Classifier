from scipy import stats
import numpy as np
from math import log;


# This method computes entropy for information gain
def entropy(class_y):

    """Compute the entropy for a list of classes

    Example:
        entropy([0,0,0,1,1,1,1,1,1]) = 0.92
    """
    # TODO: Implement this
    class_y=np.array(class_y);
    total=len(class_y);
    n0=np.count_nonzero(class_y==0);
    if(n0==0 or n0==total):
        return 0;
    else:
        p0=(n0+0.0)/total; p1=1-p0;
        return (-1)*p0*log(p0,2)-p1*log(p1,2);

def partition_classes(x, y, split_point):
    """Partition the class vector, y, by the split point. 

    Return a list of two lists where the first list contains the labels 
    corresponding to the attribute values less than or equal to split point
    and the second list contains the labels corresponding to the attribute 
    values greater than split point

    Example:
    x = [2,4,6,7,3,7,9]
    y = [1,1,1,0,1,0,0]
    split_point = 5

    output = [[1,1,1], [1,0,0,0]]
    """ 
    # TODO: Implement this
    x=np.array(x)
    y=np.array(y)
    left=y[np.where(x<=split_point)]
    right=y[np.where(x>split_point)]
    return [left,right]
    
def information_gain(previous_y, current_y):
    """Compute the information gain from partitioning the previous_classes
    into the current_classes.

    Example:
    previous_classes = [0,0,0,1,1,1]
    current_classes = [[0,0], [1,1,1,0]]

    Information gain = 0.45915
    Input:
    -----
        previous_classes: the distribution of original labels (0's and 1's)
        current_classes: the distribution of labels given a particular attr
    """
    # TODO: Implement this
    total=len(previous_y);    
    len_left=len(current_y[0]);
    if(total==0 or len_left==0 or len_left==total):
        return 0;
    pLeft=(len_left+0.0)/total; pRight=1-pLeft;
    return entropy(previous_y)-pLeft*entropy(current_y[0])- pRight*entropy(current_y[1]);

