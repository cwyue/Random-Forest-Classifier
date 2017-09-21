from util import entropy, information_gain, partition_classes
from math import sqrt


import csv
import numpy as np  # http://www.numpy.org
import ast


class DecisionTree(object):
	def __init__(self):
		self.tree = {}
		self.leaves = []

	def learn(self, X, y):
		# TODO: train decision tree and store it in self.tree
		d=X.shape[1];
		valueSet=self.possibleValues(X);
		rootNode=[];
		for i in range(d):
			rootNode.append([valueSet[i][np.argmin(valueSet[i])],valueSet[i][np.argmax(valueSet[i])]]);
		rootNode.append(0);
		node_num=0;
		nodeList=[(node_num,rootNode)];

		while(len(nodeList)!=0):
			current_num,node=nodeList.pop();
			total_list=self.domain(X,node);
			#if(len(total_list)==0):
			#	continue;
			if(entropy(y[total_list])>=0.1):
				attr,split,child_node1,child_node2=self.findBest(X[total_list,:],y[total_list],node);
				if(child_node1==node or child_node2==node):
					self.leaves.append(node);
					self.tree[current_num]=[-1,-1,-1,-1,node[9]];
					continue;
				nodeList.append((node_num+1,child_node1));
				nodeList.append((node_num+2,child_node2));
				self.tree[current_num]=[attr,split,node_num+1,node_num+2,node[9]];
				node_num=node_num+2;
			else:
				self.leaves.append(node);
				self.tree[current_num]=[-1,-1,-1,-1,node[9]];


	def classify(self, record):
		# TODO: return predicted label for a single record using self.tree
		record=np.array(record);
		d=len(record);
		node_num=0;
		while(self.tree[node_num][0]!=-1):
			if(record[self.tree[node_num][0]]<=self.tree[node_num][1]):
				node_num=self.tree[node_num][2];
			else:
				node_num=self.tree[node_num][3];
		return self.tree[node_num][4];
		'''
		for node in self.leaves:
			flag=0;
			for j in range(d):
				if (record[j]>=node[j][0] and record[j]<=node[j][1]):
					flag=flag+1;
			if(flag==d):
				return node[9];
		'''
		return 0;

	def possibleValues(self, X):
		d=X.shape[1];
		valueSet=[];
		for i in range(d):
			valueSet.append(np.unique(X[:,i]));
		return valueSet;

	def domain(self, X,node):
		d=X.shape[1];
		index=np.where( (X[:,0]>=node[0][0]) & (X[:,1]>=node[1][0]) & (X[:,2]>=node[2][0]) &
		(X[:,3]>=node[3][0]) & (X[:,4]>=node[4][0]) & (X[:,5]>=node[5][0]) & (X[:,6]>=node[6][0])
		&(X[:,7]>=node[7][0]) & (X[:,8]>=node[8][0]) & (X[:,0]<=node[0][1]) & (X[:,1]<=node[1][1])
		& (X[:,2]<=node[2][1]) & (X[:,3]<=node[3][1]) & (X[:,4]<=node[4][1]) & (X[:,5]<=node[5][1])
		& (X[:,6]<=node[6][1]) & (X[:,7]<=node[7][1]) & (X[:,8]<=node[8][1]) )[0];
		return index;

	def findBest(self, X,y,node):
		d=X.shape[1];
		m=int(sqrt(d));
		attributes=np.random.permutation(d)[:m];
		bestAttr=-1;
		bestSplit=0;
		bestGain=0;
		child_node1=list(node);
		child_node2=list(node);
		for attr in attributes:
			split=node[attr][0];
			while(split<node[attr][1]):
				current_y=partition_classes(X[:,attr], y, split);
				gain=information_gain(y, current_y);
				if(gain>bestGain):
					bestAttr=attr;
					bestSplit=split;
					bestGain=gain;
					#print bestAttr,bestSplit,"largest",bestGain
				split=split+1;
		if(bestAttr==-1):
			return -1,-1,node,node;
		child_node1[bestAttr]=[node[bestAttr][0],bestSplit];
		child_node2[bestAttr]=[bestSplit+1,node[bestAttr][1]];
		n0_left=len(self.domain(X[np.where(y==0)[0],:],child_node1));
		if(n0_left>=0.5*len(self.domain(X,child_node1))):
			child_node1[9]=0;
		else:
			child_node1[9]=1;
		n0_right=len(self.domain(X[np.where(y==0)[0],:],child_node2));
		if(n0_right>=0.5*len(self.domain(X,child_node2))):
			child_node2[9]=0;
		else:
			child_node2[9]=1;
		#print bestAttr,child_node1[bestAttr],child_node1[9],child_node2[bestAttr],child_node2[9]
		return bestAttr,bestSplit,child_node1,child_node2;

