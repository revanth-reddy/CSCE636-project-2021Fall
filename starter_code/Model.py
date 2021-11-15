### YOUR CODE HERE
# import tensorflow as tf
# import torch
import os, time
import numpy as np
from Network import MyNetwork
from ImageUtils import parse_record
import torch
import torch.nn as nn
import tqdm

"""This script defines the training, validation and testing process.
"""

class MyModel(object):

    def __init__(self, configs):
        self.configs = configs
        self.network = MyNetwork(configs)

    def model_setup(self):
        lr = self.configs['learning_rate']
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=lr,momentum=0.9, weight_decay=5e-4)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer,step_size=10,gamma=0.1)


    def train(self, x_train, y_train, configs, x_valid=None, y_valid=None):

        self.model = self.network(None,True)
        self.model_setup()

        # Determine how many batches in an epoch
        print('train_configs',configs)
        batch_size = configs['batch_size']
        max_epoch = configs['max_epochs']
        
        num_samples = x_train.shape[0]
        num_batches = num_samples // batch_size

        print('### Training... ###')
        qbar = tqdm.tqdm(range(1,max_epoch+1),position=0)
        for epoch in qbar:
            start_time = time.time()
            # Shuffle
            shuffle_index = np.random.permutation(num_samples)
            curr_x_train = x_train[shuffle_index]
            curr_y_train = y_train[shuffle_index]

            ### YOUR CODE HERE
            # Set the learning rate for this epoch
            # Usage example: divide the initial learning rate by 10 after several epochs
            
            ### YOUR CODE HERE
            val_loss_value = np.array([],dtype=np.float32)
            qbar1 = tqdm.tqdm(range(num_batches),position=1,leave=False)
            for i in qbar1:
                ### YOUR CODE HERE
                # Construct the current batch.
                # Don't forget to use "parse_record" to perform data preprocessing.
                # Don't forget L2 weight decay
                x_train_batch = curr_x_train[batch_size*i:(i+1)*batch_size,:]
                y_train_batch = curr_y_train[batch_size*i:(i+1)*batch_size]
                curr_x_batch = []
                for image in x_train_batch:
                    curr_x_batch.append(parse_record(image,True))
                curr_x_batch = np.array(curr_x_batch)
                curr_x_batch = torch.cuda.FloatTensor(curr_x_batch)
                y_train_batch = torch.cuda.LongTensor(y_train_batch)
                ### YOUR CODE HERE
                self.optimizer.zero_grad()
                ##change here--------
                outputs = self.network(curr_x_batch,True)
                # self.network.train()
                # outputs = self.network(curr_x_batch)
                loss = self.criterion(outputs,y_train_batch)
                loss.backward()
                self.optimizer.step()
                
                qbar1.set_description('Batch {:d} Num Batches {:d} Loss {:.6f}'.format(i, num_batches, loss))
                #print('Batch {:d}/{:d} Loss {:.6f}'.format(i, num_batches, loss), end='\r', flush=True)
            self.scheduler.step()
            duration = time.time() - start_time
            qbar.set_description('Epoch {:d} Loss {:.6f} Duration {:.3f} seconds.'.format(epoch, loss, duration))
            
            val_loss = self.evaluate(x_valid,y_valid)
            row = np.array([val_loss], dtype=np.float32)
            val_loss_value = np.hstack((val_loss_value, row))
            epoch_max_acc=np.argmax(val_loss_value)
            print('Maximum accuracy on val set at {:d} epoch is {:.4f}'.format(epoch_max_acc+1,val_loss_value[epoch_max_acc]))
            
            if epoch % configs['save_interval'] == 0:
                self.save(epoch)

    def evaluate(self, x, y,checkpoint_num_list=None):
        self.model = self.network(None,True)
        #checkpoint_num_list = [10,20,30,40,50,60,70,80]
        print('### Test or Validation ###')
        if checkpoint_num_list==None:
                preds = []
                for i in tqdm.tqdm(range(x.shape[0])):
                    ### YOUR CODE HERE
                    outputs = self.predict_prob(x[i])
                    _, predict = torch.max(outputs, 1)
                    preds.append(predict)
                    ### END CODE HERE
                y = torch.tensor(y)
                preds = torch.tensor(preds)
                correct_preds = torch.sum(preds==y).numpy()
                print('correct_preds',correct_preds)
                print('Validation accuracy: {:.4f}'.format(correct_preds/y.shape[0]))
        else:
            model_dir = self.configs['model_dir']
            for checkpoint_num in checkpoint_num_list:
                checkpointfile = os.path.join(model_dir, 'model-%d.ckpt'%(checkpoint_num))
                self.load(checkpointfile)

                preds = []
                for i in tqdm.tqdm(range(x.shape[0])):
                    ### YOUR CODE HERE
                    outputs = self.predict_prob(x[i])
                    _, predict = torch.max(outputs, 1)
                    preds.append(predict)
                    ### END CODE HERE
                y = torch.tensor(y)
                preds = torch.tensor(preds)
                correct_preds = torch.sum(preds==y).numpy()
                print('correct_preds',correct_preds)
                print('Test accuracy: {:.4f}'.format(correct_preds/y.shape[0]))


    def predict_prob(self, x):
        input = parse_record(x,False)
        input = np.expand_dims(input, axis=0)
        input = torch.cuda.FloatTensor(input)
        ##change here--------
        outputs = self.network(input,False)
        # self.network.eval()
        # outa = self.network(x)
        return outputs

    def save(self, epoch):
        model_dir = self.configs['model_dir']
        checkpoint_path = os.path.join(model_dir, 'model-%d.ckpt'%(epoch))
        os.makedirs(model_dir, exist_ok=True)
        torch.save(self.model.state_dict(), checkpoint_path)
        print("Checkpoint has been created.")
    
    def load(self, checkpoint_name):
        ckpt = torch.load(checkpoint_name, map_location="cpu")
        self.model.load_state_dict(ckpt, strict=True)
        print("Restored model parameters from {}".format(checkpoint_name))

### END CODE HERE