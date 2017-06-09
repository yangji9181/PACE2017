'''
	Created on Jan 23, 2016
	Handle all the data preprocessing, turning files to numpy files
	Sample graph to generate labels

	@author: Lanxiao Bai, Carl Yang
'''
import numpy as np
import ast
import utils
import random
import pickle
from scipy.sparse import csr_matrix

class Dataset(object):
	def __init__(
		self, 
		prefix='_small',
		negative=5,
		split=0.01,
		data_name='gowalla'):
		'''
			Constructor:
				data_name: Name of Dataset
		'''
		self.prefix = prefix
		self.negative = negative
		self.split = split
		self.file_path = 'data/'
		self.context_data = {}
		self.context_data['user_context'] = []
		self.context_data['spot_context'] = []
		self.generate()

	def generate(self):
		interdata_file='inter'+self.prefix+'.pkl'
		traindata_file='traindata'+self.prefix+'.pkl'
		testdata_file='testdata'+self.prefix+'.pkl'

		writeToFile = False
		try:
			f = open(self.file_path + interdata_file, 'r')
			f.close()
		except IOError:
			writeToFile = True

		if not writeToFile:	
			with open(self.file_path + interdata_file, 'rb') as f:
				inter_data = pickle.load(f)
			with open(self.file_path + traindata_file, 'rb') as f:
				self.train_data = pickle.load(f)
			with open(self.file_path + testdata_file, 'rb') as f:
				self.test_data = pickle.load(f)
			self.user_enum = inter_data['user_enum']
			self.spot_enum = inter_data['spot_enum']
			self.user_label = inter_data['user_label']
			self.spot_label = inter_data['spot_label']
			print(str(len(self.user_enum))+' users in enum loaded')
			print(str(len(self.spot_enum))+' spots in enum loaded')
			print(str(len(self.user_label))+' user context labels loaded')
			print(str(len(self.spot_label))+' spot context labels loaded')
			print(str(len(self.train_data['user']))+' training labels loaded')
			print(str(len(self.test_data['user']))+' test labels loaded')
		else:	
			inter_data = {}
			self.train_data = {}
			self.train_data['user'] = []
			self.train_data['spot'] = []
			self.train_data['label'] = []
			self.test_data={}
			self.test_data['user'] = []
			self.test_data['spot'] = []
			self.test_data['label'] = []

			self.user_enum, self.spot_enum = self.getCrossLabels()		
			self.user_dict = self.getUserGraph(self.user_enum)
			self.spot_dict = self.getSpotGraph(self.spot_enum)
			self.user_label = self.getSmoothLabels(self.user_dict)
			self.spot_label = self.getSmoothLabels(self.spot_dict)
			inter_data['user_enum'] = self.user_enum
			inter_data['spot_enum'] = self.spot_enum
			inter_data['user_label'] = self.user_label
			inter_data['spot_label'] = self.spot_label
			with open(self.file_path + interdata_file, 'wb') as f:
				pickle.dump(inter_data, f)
			print('Writing '+str(len(self.train_data['user']))+' training labels to file')
			with open(self.file_path+traindata_file, 'wb') as f:
				pickle.dump(self.train_data, f)
			print('Writing '+str(len(self.test_data['user']))+' testing labels to file')
			with open(self.file_path+testdata_file, 'wb') as f:
				pickle.dump(self.test_data, f)

	def getCrossLabels(
		self,
		file_name = 'gowalla/visited_spots.txt', 
		user_filter_lower = 100,
		spot_filter_lower = 100,
		user_filter_upper = 1000,
		spot_filter_upper = 1000
		):
		'''
			Parameter:
				file_name: File name of the file that contains the graph

			Return:
				user_enum, spot_enum
		'''
		user_dict = {}
		spot_dict = {}
		negative_sample = self.negative
		split_portion = self.split
		
		with open(self.file_path + file_name, 'r') as f:
			print('Reading file ' + file_name + ' to construct training labels')
			lines = f.readlines()
			total = len(lines)
			for line in lines:
				key = int(line.split(' ')[0])
				spots_array = ast.literal_eval(line[len(str(key)) + 1:])
				user_dict[key] = spots_array
				for spot in spots_array:
					if spot not in spot_dict:
						spot_dict[spot] = []
					spot_dict[spot].append(key)

		print('Filtering users and spots')
		for user in user_dict.keys():
			if (len(user_dict[user]) < user_filter_lower) or (len(user_dict[user]) > user_filter_upper):
				del user_dict[user]
		for spot in spot_dict.keys():
			if (len(spot_dict[spot]) < spot_filter_lower) or (len(spot_dict[spot]) > spot_filter_upper):
				del spot_dict[spot]
		print('#users:'+str(len(user_dict))+', #spots:'+str(len(spot_dict)))

		print('Generating labels')
		user_enum = {}
		spot_enum = {}
		u_counter = 0
		s_counter = 0		

		for user in user_dict.keys():
			user_enum[user] = u_counter
			u_counter += 1
			for spot in user_dict[user]:
				if spot in spot_dict:
					if spot not in spot_enum:
						spot_enum[spot] = s_counter
						s_counter += 1
					if random.random() < split_portion:
						self.train_data['user'].append(user_enum[user])
						self.train_data['spot'].append(spot_enum[spot])
						self.train_data['label'].append(1)
						for i in range(negative_sample):
							if random.random() > 0.5:
								self.train_data['user'].append(user_enum[user])
								self.train_data['spot'].append(random.randrange(len(spot_dict)))
								self.train_data['label'].append(0)
							else:
								self.train_data['user'].append(random.randrange(len(user_dict)))
								self.train_data['spot'].append(spot_enum[spot])
								self.train_data['label'].append(0)
					else:
						self.test_data['user'].append(user_enum[user])
						self.test_data['spot'].append(spot_enum[spot])
						self.test_data['label'].append(1)
						for i in range(negative_sample):
							if random.random() > 0.5:
								self.test_data['user'].append(user_enum[user])
								self.test_data['spot'].append(random.randrange(len(spot_dict)))
								self.test_data['label'].append(0)
							else:
								self.test_data['user'].append(random.randrange(len(user_dict)))
								self.test_data['spot'].append(spot_enum[spot])
								self.test_data['label'].append(0)
					
		return user_enum, spot_enum


	def getUserGraph(
		self, 
		user_enum, 
		file_name='gowalla/user_network.txt'
		):
		'''
			Parameter: 
				file_name: File name of the file that contains the graph

			Return: 
				numpy.array: uu_friend_matrix that represents the user-user friendship network

		'''

		relation_dict = {}
		print('Reading file ' + file_name + ' to construct user graph')
		density = 0
		with open(self.file_path + file_name, 'r') as f:
			lines = f.readlines()
			total = len(lines)
			for line in lines:
				key = int(line.split(' ')[0])
				if key in user_enum:
					relation_dict[user_enum[key]] = [user_enum[i] for i in ast.literal_eval(line[len(str(key)) + 1:]) if i in user_enum]
					density += len(relation_dict[user_enum[key]])
		density = density * 1.0 / (len(user_enum)*len(user_enum))
		print('Density of user graph: '+str(density))
		
		return relation_dict


	def getSpotGraph(
		self, 
		spot_enum,
		sample_portion = 0.01,
		sample_radius = 0.5,
		file_name='gowalla/spot_location.txt'):
		'''
			Parameter:
				file: File that contains spot ids and latitudes and longitudes.
				radius: The maximum distances for two locations to be connected
				dict: spot_enum that records each spot id's correspondent number computed by getVisitedGraph

			Return: 
				numpy.array: ss_location_matrix that represents the spot-spot location network
				list of pairs: ss_location_label that represents labels generated from the spot-spot location network
			
		'''

		coordinates = {}
		with open(self.file_path + file_name, 'r') as f:
			print('Reading file ' + file_name + ' to construct spot graph')

			lines = f.readlines()
			total = len(lines)
			for line in lines:
				#print("Loading:" + str(counter) + "/" + str(total - 1) + "--" + line)
				splited = line.split(' ')
				n = 0
				for i in splited:
					if i == 'null':
						n = 1
				if n == 0:
					splited = [float(i) for i in line.split(' ')]
					spot, x, y = splited
					spot = int(spot)
					if spot in spot_enum:
						coordinates[spot_enum[spot]] = (x, y)

		relation_dict = {}
		density = 0
		sample_size = int(len(spot_enum)*sample_portion)
		print('Sampling '+str(sample_size)+' base spots to build spot graph')
		base_points = random.sample(coordinates.keys(), k=sample_size)
		for base in base_points:
			#print("Loading:" + str(s_counter) + "/" + str(self.sample_size))
			cell = []
			for i in coordinates.keys():
				if utils.distance(coordinates[i], coordinates[base]) < sample_radius:
					cell.append(i)
			#print('Cell '+str(base)+' has '+str(len(cell))+' spots')
			for i in cell:
				for j in cell:
					if i != j:
						if i not in relation_dict:
							relation_dict[i] = set()
						if j not in relation_dict:
							relation_dict[j] = set()
						relation_dict[i].add(j)
						relation_dict[j].add(i)

		for i in relation_dict.keys():
			relation_dict[i] = list(relation_dict[i])
			density += len(relation_dict[i])
		density = density * 1.0 / (len(spot_enum)*len(spot_enum))
		print('Density of spot graph: '+str(density))
		return relation_dict


	def getSmoothLabels(
		self,
		graph_dict,
		path_portion=0.01,
		path_length=10,
		samples_num=5,
		window_size=3):
		'''
			Parameter:
				graph_dict: Dict that stores the graph

			Return:
				Labels sampled from the graph
		'''

		print('Generating smooth labels')
		labels = {}
		path_num = int(len(graph_dict)*path_portion)
		for i in range(path_num):
			path = []
			for j in range(path_length):
				if len(path) == 0:
					path.append(graph_dict.keys()[random.randrange(len(graph_dict))])
				else:
					if path[len(path)-1] not in graph_dict or len(graph_dict[path[len(path)-1]]) == 0:
						break
					cands = graph_dict[path[len(path)-1]]
					path.append(cands[random.randrange(len(cands))])
			if len(path) > 1:
				for k in range(samples_num):
					while True:
						tup = random.sample(path, k=2)
						if abs(path.index(tup[0]) - path.index(tup[1])) < window_size:
							break
					if tup[0] not in labels:
						labels[tup[0]] = []
					if tup[1] not in labels[tup[0]]:
						labels[tup[0]].append(tup[1])

		return labels

	def generateContextLabels(self):
		print('Generating '+str(len(self.train_data['label']))+' context labels')
		for i in range(len(self.train_data['label'])):
			print(str(i))
			tmp = [0] * len(self.user_enum)
			user = self.train_data['user'][i]
			if user in self.user_label:
				user_context = self.user_label[user]	
				for j in user_context:
					tmp[j] = 1
			self.context_data['user_context'].append(np.array(tmp))
			
			tmp = [0]*len(self.spot_enum)
			row_ind = []
			col_ind = []
			data = []
			tmp = [0] * len(self.spot_enum)
			spot = self.train_data['spot'][i]
			if spot in self.spot_label:
				spot_context = self.spot_label[spot]
				for j in spot_context:
					tmp[j] = 1
			self.context_data['spot_context'].append(np.array(tmp))

	def getContextLabels(self):
		self.generateContextLabels()
		return self.context_data


if __name__ == "__main__":
	dt = Dataset()
