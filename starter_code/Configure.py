# Below configures are examples, 
# you can modify them as you wish.

### YOUR CODE HERE

model_configs = {
	"name": 'MyModel',
	"save_dir": '../saved_models/v1/',
	"depth": 2,
	"resnet_version":1,
	"learning_rate":0.1,
	"model_dir": '../saved_models/v1/',
	"num_classes":10,
	"first_num_filters":16,
 	"resnet_size":18
	# ...
}

training_configs = {
	"learning_rate": 0.01,
	"model_dir": '../saved_models/v1/',
	"max_epochs":50,
	"batch_size":128,
	"save_interval":5
	# ...
}

### END CODE HERE