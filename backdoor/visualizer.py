import torch 
import torch.nn as nn 
import numpy as np
import torch.optim as optim

class repeat_iter:
    def __init__(self, dataloader):
        self.dataloader = dataloader
        self.iter = iter(self.dataloader)

    def next(self):
        try:
            x = next(self.iter) 
        except StopIteration:
            self.iter = iter(self.dataloader)
            x = next(self.iter) 
        return x

class Visualizer:
    def __init__ (self,model,args):
        self.model = model
        self.regularization = args.regularization
        self.init_cost = args.init_cost
        self.steps = args.step
        self.lr = args.lr
        self.num_classes = args.num_classes
        self.attack_succ_threshold = args.attack_succ_threshold
        self.patience = args.patience
        self.channels = args.channels
        self.batch_size = args.batch_size
        self.mask_size = [args.input_width,args.input_height]
        self.pattern_size = [args.channels,args.input_width,args.input_height]
        self.device = torch.device("cuda:%d" % args.device)
        self.epsilon = args.epsilon 
        self.cost_multiplier = args.cost_multiplier
        self.cost_multiplier_up = args.cost_multiplier
        self.cost_multiplier_down = args.cost_multiplier ** 1.5
        self.early_stop = args.early_stop
        self.early_stop_threshold = args.early_stop_threshold
        self.early_stop_patience = args.early_stop_patience


        mask = torch.zeros(self.mask_size).to(self.device)
        pattern = torch.zeros(self.pattern_size).to(self.device)

        self.mask_tanh_tensor = torch.zeros_like(mask).to(self.device)
        self.pattern_tanh_tensor = torch.zeros_like(pattern).to(self.device)
        

        mask_tensor_unrepeat = (torch.tanh(self.mask_tanh_tensor) / (2 - self.epsilon) + 0.5)
        self.mask_tensor = mask_tensor_unrepeat.repeat(self.channels,1,1)
    
        self.pattern_raw_tensor = (torch.tanh(self.pattern_tanh_tensor) / (2 - self.epsilon) + 0.5)
    
    def reset_state(self,pattern_init,mask_init):
        self.cost = self.init_cost
        self.cost_tensor = self.cost
        

        mask_np = mask_init.cpu().numpy()
        mask_tanh = np.arctanh((mask_np - 0.5) * (2-self.epsilon))
        mask_tanh = torch.from_numpy(mask_tanh).to(self.device)

        pattern_np = pattern_init.cpu().numpy()
        pattern_tanh = np.arctanh((pattern_np - 0.5) * (2 - self.epsilon))
        pattern_tanh = torch.from_numpy(pattern_tanh).to(self.device)
        
        self.mask_tanh_tensor = mask_tanh
        self.mask_pattern_tensor = pattern_tanh

        self.mask_tanh_tensor.requires_grad = True
        self.pattern_tanh_tensor.requires_grad = True
        
        #return mask_tanh, pattern_tanh

    def update_tensor(self,mask_tanh_tensor,pattern_tanh_tensor):
        
        mask_tensor_unrepeat = (torch.tanh(mask_tanh_tensor) / (2 - self.epsilon) + 0.5)
        self.mask_tensor = mask_tensor_unrepeat.repeat(self.channels,1,1)
        self.pattern_raw_tensor = (torch.tanh(pattern_tanh_tensor) / (2 - self.epsilon) + 0.5)

    def visualize(self,data_loader,y_target,pattern_init,mask_init):
        
        #mask_tanh, pattern_tanh = self.reset_state(pattern_init,mask_init)
        self.reset_state(pattern_init,mask_init)

        #mask_tanh.requires_grad = True
        #pattern_tanh.requires_grad = True
        #print(mask_tanh.requires_grad)
        #permute = [2,1,0]

        best_mask = None
        best_pattern = None 
        best_reg = float('inf')

        log = []
        cost_set_counter = 0
        cost_down_counter = 0
        cost_up_counter = 0
        cost_up_flag = False
        cost_down_flag = False
        early_stop_counter = 0
        early_stop_reg_best = best_reg


        y_target_tensor = torch.Tensor([y_target]).long().to(self.device)
        
        #pattern_tanh_tensor = self.pattern_tanh_tensor
        #mask_tanh_tensor = self.mask_tanh_tensor
        
        
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam([self.pattern_tanh_tensor,self.mask_tanh_tensor],lr=self.lr,betas=(0.5, 0.9))
        #loader = repeat_iter(data_loader)
        loader = data_loader
        for step in range(self.steps):

            loss_ce_list = []
            loss_reg_list = []
            loss_list = []
            loss_acc_list = []
            for idx in range(1):
                img,name = next(loader)
                img = img.to(self.device)
                #img = img[:,permute,:,:]

                Y_target = y_target_tensor.repeat(img.size()[0])
                #mask_tensor_unrepeat = (torch.tanh(mask_tanh) / (2 - self.epsilon) + 0.5)
                #mask_tensor = mask_tensor_unrepeat.repeat(self.channels,1,1)
                #pattern_raw_tensor = (torch.tanh(pattern_tanh) / (2 - self.epsilon) + 0.5)
                self.update_tensor(self.mask_tanh_tensor,self.pattern_tanh_tensor)
                X_adv_tensor = (1-self.mask_tensor) * img + self.mask_tensor * self.pattern_raw_tensor
                 

                optimizer.zero_grad()
                

                output_tensor = self.model(X_adv_tensor)

                pred = output_tensor.argmax(dim=1, keepdim=True) 
                
                self.loss_acc = pred.eq(Y_target.long().view_as(pred)).sum().item() / (img.size()[0])

                self.loss_ce = criterion(output_tensor,Y_target)
                
                self.loss_reg = torch.sum(torch.abs(self.mask_tensor)) / self.channels

                self.loss = self.loss_ce + self.loss_reg * self.cost_tensor 

                self.loss.backward()
                optimizer.step()
                
                        
                #mask_tensor_unrepeat = (torch.tanh(mask_tanh_tensor) / (2 - self.epsilon) + 0.5)
                #mask_tensor = mask_tensor_unrepeat.repeat(self.channels,1,1)
                #pattern_raw_tensor = (torch.tanh(pattern_tanh_tensor) / (2 - self.epsilon) + 0.5)

                print('Target: {}, Idx: {}, Step: {}/{}, Loss: {:.4f}, Acc: {:.2f}%, CE_Loss: {:.4f}, Reg_Loss:{:.4f}'.format(
                y_target, idx, step, self.steps, self.loss,self.loss_acc * 100,self.loss_ce,self.loss_reg)
                )
                loss_ce_list.append(self.loss_ce.item())
                loss_reg_list.append(self.loss_reg.item())
                loss_list.append(self.loss.item())
                loss_acc_list.append(self.loss_acc)

            avg_loss_ce = np.mean(loss_ce_list)
            avg_loss_reg = np.mean(loss_reg_list)
            avg_loss = np.mean(loss_list)
            avg_loss_acc = np.mean(loss_acc_list)
            print(avg_loss_acc)
            if avg_loss_acc >= self.attack_succ_threshold and avg_loss_reg < best_reg:
                best_mask = self.mask_tensor
                print('best_mask update')
                best_pattern = self.pattern_raw_tensor
                best_reg = avg_loss_reg
            
            if self.early_stop:
                print(early_stop_counter)
                print(cost_up_counter)
                print(cost_down_counter)
                if best_reg < float('inf'):
                    if best_reg >= self.early_stop_threshold * early_stop_reg_best and avg_loss_acc >= self.attack_succ_threshold:
                        early_stop_counter +=1
                    else:
                        early_stop_counter = 0
                early_stop_reg_best = min(best_reg,early_stop_reg_best)

                if (cost_down_flag and cost_up_flag and early_stop_counter > self.early_stop_patience):
                    print('early stop')
                    break

            if self.cost == 0 and avg_loss_acc >= self.attack_succ_threshold:
                cost_set_counter += 1
                if cost_set_counter >= self.patience:
                    self.cost = self.init_cost
                    self.cost_tensor =  self.cost
                    cost_up_counter = 0
                    cost_down_counter = 0
                    cost_up_flag = False
                    cost_down_flag = False
            else:
                cost_set_counter = 0

            if avg_loss_acc >= self.attack_succ_threshold:
                cost_up_counter += 1
                cost_down_counter = 0
            else:
                cost_up_counter = 0
                cost_down_counter += 1

            if cost_up_counter >= self.patience:
                cost_up_counter = 0
                self.cost *= self.cost_multiplier_up
                self.cost_tensor =  self.cost
                cost_up_flag = True
            elif cost_down_counter >= self.patience:
                cost_down_counter = 0
                self.cost /= self.cost_multiplier_down
                self.cost_tensor =  self.cost
                cost_down_flag = True
            
        if  best_mask is None:
            best_mask = self.mask_tensor
            print('best_mask update')
            best_pattern = self.pattern_raw_tensor
            best_reg = avg_loss_reg

        return best_pattern, best_mask


