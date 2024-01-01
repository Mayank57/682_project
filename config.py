class Config:
	def __init__(self, labels_dict = {'accessoryNothing':0,'hairLong':1,'upperBodyLongSleeve':2,'upperBodyCasual':3,'upperBodyThinStripes':4,
             'upperBodyThickStripes':4,'upperBodyOther':5,'lowerBodyCasual':6,'lowerBodyTrousers':7,
             'lowerBodyLongSkirt':8,'lowerBodyShortSkirt':8,'footwearSneaker':9,'carryingNothing':10,
             'personalLess15':11,'personalLess30':11,'personalLess45':12,'personalLess60':12,'personalMale':13},
             attributes = ['accessoryNothing', 'hairLong', 'upperBodyLongSleeve', 'upperBodyCasual', 'upperBodyThinStripes', 'upperBodyOther', 'lowerBodyCasual', 'lowerBodyTrousers', 'lowerBodyLongSkirt', 'footwearSneaker', 'carryingNothing', 'personalLess30', 'personalLess60', 'personalMale'],
             group_indices_list = [[1],[2,3,4,5],[6,7,8],[9],[0,10,11,12,13]], 
             num_groups = 5, momentum = 0.9, num_cascades = 3, num_epochs = 1, batch_size = 16, learning_rate=0.0005, dataset_length = 4562, 
             path = '/Users/mayank57/Downloads/PETA/PETA dataset/', checkpoint_path = 'training_tp1/', plot_save_path = 'plots/'):
		self.labels_dict = labels_dict
		self.num_attributes = len(labels_dict)  # Change as per your dataset
		self.num_groups = num_groups       # Change as per your dataset
		self.momentum = momentum       # Hyperparameter
		self.num_cascades = num_cascades     # Number of cascaded SSCA modules
		self.num_epochs = num_epochs
		self.attributes = attributes
		self.group_indices_list = group_indices_list
		self.batch_size = batch_size 
		self.learning_rate = learning_rate
		self.dataset_length = dataset_length
		self.path = path
		self.checkpoint_path = checkpoint_path
		self.plot_save_path = plot_save_path