#!/usr/bin/env python
# coding: utf-8

# In[4]:

# sampler 
# margin loss


import numpy as np
import itertools as it
import torch
import random

# In[6]:

def select_loss(loss, sampling_method, metric,e_size,num_cls,lambda_,    lr,wd,momentum, to_optim):

	"""
	Args:
		loss:    name of the loss
		metric: Loss(metric,e_size)   % num_cls
		optim : rm or Maha : L -> to_optim
		opt:      
			margin
			metric
			sampler 
			...
		to_optim: list of trainable parameters. Is extend if loss function contains those as well.

	Returns:
		criterion (torch.nn.Module inherited), to_optim (optionally appended)

	"""

	if loss=='triplet':

		criterion = TripletLoss(metric=metric,sampling_method=sampling_method,e_size = e_size,lambda_=lambda_) #***
		if metric =='rM' or metric == 'maha':
			to_optim    += [{'params':criterion.parameters(), 'lr': lr,'momentum':momentum, 'weight_decay':wd}]
			
	elif loss=='margin':
		criterion = MarginLoss(metric=metric,sampling_method=sampling_method,e_size=e_size,n_classes=num_cls,lambda_=lambda_)
		to_optim    += [{'params':criterion.parameters(), 'lr': lr,'momentum':momentum, 'weight_decay':wd}]
		
	elif loss=='lifted':

		criterion    = LiftedLoss(metric = metric,sampling_method=sampling_method,e_size=e_size,lambda_=lambda_)
		if metric =='rM' or metric == 'maha':
			to_optim    += [{'params':criterion.parameters(), 'lr': lr,'momentum':momentum, 'weight_decay':wd}]
		
		
	else:
		raise Exception('Loss {} not available!'.format(loss))
	return criterion, to_optim


# In[3]:


class Sampler():
	'''
	rand
	dist: 
	lifted
	'''
	def __init__(self,method):
		
		if method == 'rand':
			self.give = self.rand_sampler
		elif method == 'dist':
			self.give = self.dist_sampler
		elif method == 'lifted':
			self.give = self.lifted_sampler
		else:
			raise Exception('Sampler {} not available!'.format(method))
	
	def rand_sampler(self, batch, labels):
		"""
		This methods finds all available triplets in a batch given by the classes provided in labels, and randomly
		selects <len(batch)> triplets.
		Args:
			batch:  np.ndarray or torch.Tensor, batch-wise embedded training samples.
			labels: np.ndarray or torch.Tensor, ground truth labels corresponding to batch.
		Returns:
			list of sampled data tuples containing reference indices to the position IN THE BATCH.
		"""
		if isinstance(labels, torch.Tensor): labels = labels.view(len(labels)).cpu().detach().numpy()
		#unique_classes = np.unique(labels)
		label_set, count = np.unique(labels, return_counts=True)

		unique_classes  = label_set[count>=2]
		
		#print(unique_classes)
		indices        = np.arange(len(batch))
		class_dict     = {i:indices[labels==i] for i in unique_classes}

		sampled_triplets = [list(it.product([x],[x],[y for y in unique_classes if x!=y])) for x in unique_classes]
		
		sampled_triplets = [x for y in sampled_triplets for x in y] # unfolded
        
		#from cls to idx and unfolded
		sampled_triplets = [[x for x in list(it.product(*[class_dict[j] for j in i])) if x[0]!=x[1]] for i in sampled_triplets]
		sampled_triplets = [x for y in sampled_triplets for x in y]

		#NOTE: The number of possible triplets is given by #unique_classes*(2*(samples_per_class-1)!)*(#unique_classes-1)*samples_per_class
		#print('Sample values:')
		#print(sampled_triplets, batch.shape[0])
		sampled_triplets = random.sample(sampled_triplets, batch.shape[0])
		return sampled_triplets


	
	
	
	def dist_sampler(self, batch, labels, lower_cutoff=0.4, upper_cutoff=1.8):

		"""

		This methods finds all available triplets in a batch given by the classes provided in labels, and select

		triplets based on distance sampling introduced in 'Sampling Matters in Deep Embedding Learning'.



		Args:

			batch:  np.ndarray or torch.Tensor, batch-wise embedded training samples.

			labels: np.ndarray or torch.Tensor, ground truth labels corresponding to batch.

			lower_cutoff:This will affect the dist bet a_n
			upper_cutoff: float, upper cutoff value for positives that are too far away from the anchor embeddings. Set to literature value. They will be assigned a zero-sample probability.

		Returns:
			bs* [a,p,n]

			list of sampled data tuples containing reference indices to the position IN THE BATCH.

		"""

		if isinstance(labels, torch.Tensor): labels = labels.view(len(labels)).detach().cpu().numpy()
		
		label_set, count = np.unique(labels, return_counts=True)

		label_set  = label_set[count>=2]
		
		
		bs = batch.shape[0]
		distances    = self.pdist(batch.detach()).clamp(min=lower_cutoff,max= upper_cutoff)
		#print(torch.max(distances),torch.min(distances),torch.sum(batch)/batch.size(0)) #**

		positives, negatives = [],[]
		labels_visited = []
		anchors = []
		
		for i in range(bs):
			if labels[i] not in label_set:
				continue
			neg = labels!=labels[i]; 
			pos = labels==labels[i]
			
			q_d_inv = self.inverse_sphere_distances(batch, distances[i], labels, labels[i])
			
			anchors.append(i)
			
			#Sample positives randomly
			pos[i] = 0 # 去掉自己 
			positives.append(np.random.choice(np.where(pos)[0]))   # one sample is returned randomly
			
			#Sample negatives by distance  
			negatives.append(np.random.choice(bs,p=q_d_inv))  # 0~bs  
			# 存疑https://docs.scipy.org/doc/numpy-1.13.0/reference/generated/numpy.random.choice.html
			

		sampled_triplets = [[a,p,n] for a,p,n in zip(anchors, positives, negatives)]

		return sampled_triplets
	
	
	def lifted_sampler(self, batch, labels, lower_cutoff=0.4, upper_cutoff=1.8):

		"""

		This methods finds all available triplets in a batch given by the classes provided in labels, and select

		triplets based on distance sampling introduced in 'Sampling Matters in Deep Embedding Learning'.



		Args:

			batch:  np.ndarray or torch.Tensor, batch-wise embedded training samples.

			labels: np.ndarray or torch.Tensor, ground truth labels corresponding to batch.

			lower_cutoff: float, lower cutoff value for negatives that are too close to anchor embeddings. Set to literature value. They will be assigned a zero-sample probability.

			upper_cutoff: float, upper cutoff value for positives that are too far away from the anchor embeddings. Set to literature value. They will be assigned a zero-sample probability.

		Returns:
			bs* [a,p,n]

			list of sampled data tuples containing reference indices to the position IN THE BATCH.

		"""

		if isinstance(labels, torch.Tensor): labels = labels.view(len(labels)).detach().cpu().numpy()

		label_set, count = np.unique(labels, return_counts=True)
		label_set  = label_set[count>=2]
		bs = batch.shape[0]
		distances    = self.pdist(batch.detach()).clamp(min=lower_cutoff,max= upper_cutoff) # for probability
		dists = self.pdist(batch.detach()).detach().cpu().numpy() # for comparison

		positives, negatives = [],[]
		labels_visited = []
		anchors = []

		for i in range(bs):
			if labels[i] not in label_set: # to eliminate the sample which do not have pot in the batch
				continue
			neg = labels!=labels[i]
			pos = labels==labels[i]
            
			d = dists[i] 
			q_d_inv = self.inverse_sphere_distances(batch, distances[i], labels, labels[i])

			anchors.append(i)
			
			#Sample positives randomly
			pos[i] = 0 # delete itself
			pos_mask = np.logical_and(pos,d>d[np.where(neg)[0]].min())
			if pos_mask.sum()>0: # if exist sample meet the criterion 
				positives.append(np.random.choice(np.where(pos_mask)[0]))   # boolen 2 indices 如果有，就从里面挑一个
			else:
				positives.append(np.random.choice(np.where(pos)[0]))
                
			#positives.append(np.random.choice(np.where(pos)[0]))   # one sample is returned，
			
			#Sample negatives by distance
			neg_mask = np.logical_and(neg,d<d[np.where(pos)[0]].max())
			size = neg_mask.sum()
			if size< 20:
				size = 20
			
			negatives.append(np.random.choice(bs,size,p=q_d_inv))  # 0~bs  
			# 存疑https://docs.scipy.org/doc/numpy-1.13.0/reference/generated/numpy.random.choice.html
			

		sampled_lifted = [[a,p,*n] for a,p,n in zip(anchors, positives, negatives)]

		return sampled_lifted
	
		 
	def pdist(self, A, eps = 1e-4):

		"""

		Efficient function to compute the distance matrix for a matrix A.



		Args:

			A:   Matrix/Tensor for which the distance matrix is to be computed.

			eps: float, minimal distance/clampling value to ensure no zero values.

		Returns:

			distance_matrix, clamped to ensure no zero values are passed.

		"""

		prod = torch.mm(A, A.t())  # size: bs*bs

		norm = prod.diag().unsqueeze(1).expand_as(prod) # diag() :return the diagonal elem of the inputs
		# unsqueeze(1): 把一维向量扩展成2维 ： n*1
		# expand_as

		res = (norm + norm.t() - 2 * prod).clamp(min = 0)  # (a^2+b^2 - 2ab) = (a-b)^2

		return res.clamp(min = eps).sqrt()
		 
		 
	def inverse_sphere_distances(self, batch, dist, labels, anchor_label):

		"""

		Function to utilise the distances of batch samples to compute their

		probability of occurence, and using the inverse to sample actual negatives to the resp. anchor.



		Args:

			batch:        torch.Tensor(), batch for which the sampling probabilities w.r.t to the anchor are computed. Used only to extract the shape.

			dist:         torch.Tensor(), computed distances between anchor to all batch samples.

			labels:       np.ndarray, labels for each sample for which distances were computed in dist.

			anchor_label: float, anchor label

		Returns:

			distance_matrix, clamped to ensure no zero values are passed.

		"""

		bs,dim       = len(dist),batch.shape[-1]
		#negated log-distribution of distances of unit sphere in dimension <dim>
		log_q_d_inv = ((2.0 - float(dim)) * torch.log(dist) - (float(dim-3) / 2) * torch.log(1.0 - 0.25 * (dist.pow(2))))

		#Set sampling probabilities of positives to zero

		#log_q_d_inv[np.where(labels==anchor_label)[0]] = 0
		#q_d_inv = log_q_d_inv #**
		q_d_inv     = torch.exp(log_q_d_inv - torch.max(log_q_d_inv)) # - max(log) for stability
		#Set sampling probabilities of positives to zero

		q_d_inv[np.where(labels==anchor_label)[0]] = 0



		### NOTE: Cutting of values with high distances made the results slightly worse.

		# q_d_inv[np.where(dist>upper_cutoff)[0]]    = 0



		#Normalize inverted distance for probability distr.

		q_d_inv = q_d_inv/q_d_inv.sum()

		return q_d_inv.detach().cpu().numpy()



		


# In[ ]:


class Metric():
	
	'''
	
	'''
	
	def __init__(self,method):
		
		if method == 'E':
			self.give = self.e_
		elif method == 'snr':
			self.give = self.snr_
		elif method == 'rE':
			self.give = self.re_
		elif method == 'maha':
			self.give = self.m_
		elif method == 'rM':
			self.give = self.rm_
		else:
			raise Exception('Metric {} not available!'.format(method))
	
	def e_(sefl,anchor,b,cov):
		"""
		Return: 
			d
		"""
		d = torch.norm(anchor-b)
		return d
	
	def snr_(self,anchor,b,cov):
		'''
		Args:
			anchor: t
			b: pos or neg
			
		returns:
			d_s
		'''
		tmp =anchor-b
		upper = self.var_(tmp)
		lower = self.var_(anchor)
		d  = upper/lower
		return d
	
	def var_(self,a):
		mean = a.mean()
		return torch.norm(a-mean).pow(2)/ len(a)
	
	def re_(self,anchor,b,cov):
		'''
		
		return:
			d_s = d_E^2(hi,hj) / d_E^2(hi)
		'''
		
		upper = torch.norm(anchor-b)
		lower = torch.norm(anchor)
		
		
		d = upper/lower
		return d.pow(2)
	
	
	
	def m_(self,anchor,b,cov):
		'''
		L  = torch.ones(m)* sqrt(1/m)
		http://jmlr.csail.mit.edu/papers/volume10/weinberger09a/weinberger09a.pdf
		Args:
			anchor,b (1*m)
			
			?? size of the output
		
		'''
		#anchor.view(-1,m) # L:1*n
		left = (anchor -b).view(1,-1)  
		right = torch.matmul(cov,left.t())
		
		d = torch.matmul(left,right)
		return d
	
	def rm_(self,anchor,b,cov):
		upper = self.m_(anchor,b,cov)
		
		right = torch.matmul(cov,anchor.view(-1,1)) # anchor:1*m , an.t(): m*1
		lower = torch.matmul(anchor.view(1,-1),right)
		
		d = upper / lower
		
		return d
		


# In[ ]:



### Standard Triplet Loss, finds triplets in Mini-batches.
#____________________________________________________________________________
class LiftedLoss(torch.nn.Module):

	def __init__(self, e_size , lambda_,sampling_method,metric  ,      margin=1.):

		"""
		Args:
			margin:             float, Triplet Margin - Ensures that positives aren't placed arbitrarily close to the anchor.

								Similarl, negatives should not be placed arbitrarily far away.

			sampling_method:    Method to use for sampling training triplets. Used for the Sampler-class.

		"""

		super(LiftedLoss, self).__init__()

		self.margin  = margin
		self.sampler   = Sampler(method=sampling_method)
		self.dist = Metric(method = metric)
		self.metric_method = metric
		self.e_size = e_size
		self.lambda_ = lambda_
		if metric=='rM' or metric =='maha':
			self.L = torch.nn.Parameter(torch.ones((1,e_size))/e_size)


	def forward(self, batch, labels):

		"""
		Args:
			batch:   torch.Tensor() [(BS x embed_dim)], batch of embeddings
			labels:  np.ndarray [(BS x 1)], for each element of the batch assigns a class [0,...,C-1]
		Returns:
			triplet loss (torch.Tensor(), batch-averaged)

		"""

		#Sample triplets to use for training.
		
		# cal the cov
		if self.metric_method == 'maha' or self.metric_method == 'rM':
			cov = torch.mm(self.L.t(),self.L)
		else:
			cov =torch.tensor(0.).cuda()
		

		sampled_lifted = self.sampler.give(batch, labels)  # bs* 2+N

		#Compute triplet loss
		
		d_list = [] 
		for Npair in sampled_lifted:  # Npair = 
			d_an = torch.tensor(0.).type(torch.FloatTensor)
			d_pn = torch.tensor(0.).type(torch.FloatTensor)
			if torch.cuda.is_available():
				d_an = d_an.cuda()
				d_pn  = d_pn.cuda()
			anchor = batch[Npair[0],:]   # feature of anchor
			pos = batch[Npair[1],:]     # feature vector of positive
			
			d_ap = self.dist.give(anchor,pos,cov) # cal the dist bet anchor and pos
			
			for neg_idx in Npair[3:]: 
				 
				neg = batch[neg_idx,:]
				d_an += torch.exp(self.margin-self.dist.give(anchor,neg,cov)) # exp(margin - d_an)
				d_pn += torch.exp(self.margin-self.dist.give(pos,neg,cov))
			d = torch.nn.functional.relu(d_ap+torch.log(d_an+d_pn))
			d_list.append(d)
			
		loss =torch.stack(d_list)
		
		return torch.mean(loss)*0.5+self.lambda_*torch.sum(torch.abs(torch.sum(batch,1)))/batch.size(0)  # 1/ (2*P) Sum(...) P: num_of_pair = batch_size   ;mean()= 1/P * Sum[]




# In[ ]:
#___________________________________________________________________________________________________________

class MarginLoss(torch.nn.Module):

	def __init__(self, e_size,n_classes,lambda_,metric ,margin=0.5, nu=0, beta=1.2, beta_constant=False, sampling_method='dist'):

		"""

		Basic Margin Loss as proposed in 'Sampling Matters in Deep Embedding Learning'.

		Args:

			margin:          float, fixed triplet margin (see also TripletLoss).

			nu:              float, regularisation weight for beta. Zero by default (in literature as well).

			beta:            float, initial value for trainable class margins. Set to default literature value.

			n_classes:       int, number of target class. Required because it dictates the number of trainable class margins.

			beta_constant:   bool, set to True if betas should not be trained.

			sampling_method: str, sampling method to use to generate training triplets.

		Returns:

			Nothing!

		"""

		super(MarginLoss, self).__init__()

		self.margin             = margin
		self.e_size = e_size
		self.n_classes          = n_classes
		self.beta_constant = beta_constant
		self.beta     = beta if beta_constant else torch.nn.Parameter(torch.ones(n_classes)*beta)
		self.lambda_ = lambda_
		self.nu                 = nu

		self.sampler   = Sampler(method=sampling_method)
		self.dist = Metric(method = metric)
		self.metric_method = metric       
		if metric=='rM' or metric =='maha':
			self.L = torch.nn.Parameter(torch.ones((1,e_size))/e_size) 

	def forward(self, batch, labels):

		"""
		Args:
			batch:   torch.Tensor() [(BS x embed_dim)], batch of embeddings
			labels:  np.ndarray [(BS x 1)], for each element of the batch assigns a class [0,...,C-1]

		Returns:
			margin loss (torch.Tensor(), batch-averaged)
		"""
		if self.metric_method == 'maha' or self.metric_method == 'rM':
			cov = torch.mm(self.L.t(),self.L)
		else:
			cov = torch.tensor(0.).cuda()


		if isinstance(labels, torch.Tensor): labels = labels.view(len(labels)).cpu().detach().numpy()#.astype('int')
		sampled_triplets = self.sampler.give(batch, labels)
		#Compute distances between anchor-positive and anchor-negative.
		d_ap, d_an = [],[]

		for triplet in sampled_triplets:

			train_triplet = {'Anchor': batch[triplet[0],:], 'Positive':batch[triplet[1],:], 'Negative':batch[triplet[2]]}

			pos_dist = self.dist.give(train_triplet['Anchor'],train_triplet['Positive'],cov)   

			neg_dist = self.dist.give(train_triplet['Anchor'],train_triplet['Negative'],cov)

			d_ap.append(pos_dist)

			d_an.append(neg_dist)

		d_ap, d_an = torch.stack(d_ap), torch.stack(d_an)



		#Group betas together by anchor class in sampled triplets (as each beta belongs to one class).

		if self.beta_constant:
			beta = self.beta
		else:

			beta = torch.stack([self.beta[labels[triplet[0]]] for triplet in sampled_triplets]).type(torch.FloatTensor)
			if torch.cuda.is_available():
				beta = beta.cuda()



		#Compute actual margin postive and margin negative loss

		pos_loss = torch.nn.functional.relu(d_ap-beta+self.margin)

		neg_loss = torch.nn.functional.relu(beta-d_an+self.margin)

		#Compute normalization constant
		pair_count = torch.sum((pos_loss>0.)+(neg_loss>0.)).type(torch.FloatTensor)
		if torch.cuda.is_available():
			pair_count = pair_count.cuda()
		# ** I think pair_count = batch_size,

		#Actual Margin Loss
		loss = torch.sum(pos_loss+neg_loss) if pair_count==0. else torch.sum(pos_loss+neg_loss)/pair_count

		#(Optional) Add regularization penalty on betas.

		

		return loss+ self.lambda_*torch.sum(torch.abs(torch.sum(batch,1)))/batch.size(0)



# In[ ]:
#____________________________________________________________________________

class TripletLoss(torch.nn.Module):

	def __init__(self, e_size,lambda_,metric , margin=1, sampling_method='dist'):
		"""
		Basic Triplet Loss as proposed in 'FaceNet: A Unified Embedding for Face Recognition and Clustering'
		Args:
			margin:             float, Triplet Margin - Ensures that positives aren't placed arbitrarily close to the anchor.
								Similarl, negatives should not be placed arbitrarily far away.
			sampling_method:    Method to use for sampling training triplets. Used for the Sampler-class.
		"""
		super(TripletLoss, self).__init__()
		
		self.margin             = margin
		self.sampler            = Sampler(method=sampling_method)
		self.metric_method = metric
		self.dist = Metric(method = metric)
		self.lambda_ = lambda_
		self.e_size = e_size
		if metric=='rM' or metric =='maha':
			self.L = torch.nn.Parameter(torch.ones((1,e_size))/e_size) 

	def triplet_distance(self, anchor, positive, negative,cov):

		"""
		Compute triplet loss.
		Args:

			anchor, positive, negative: torch.Tensor(), resp. embeddings for anchor, positive and negative samples.

		Returns:
			triplet loss (torch.Tensor())

		"""
		d_ap = self.dist.give(anchor,positive,cov)
		d_an = self.dist.give(anchor,negative,cov)
		#print('mean of anchor',torch.mean(anchor))
		#if d_ap<0: print('D_ap',d_ap)
		#if d_an<0: print('D_an',d_an)
		if d_ap<0 or d_an <0:
			print('+1',d_ap,d_an)
			return -torch.nn.functional.relu(torch.abs(d_ap))
		return torch.nn.functional.relu( d_ap - d_an + self.margin )
	
	def forward(self, batch, labels):

		"""
		Args:

			batch:   torch.Tensor() [(BS x embed_dim)], batch of embeddings

			labels:  np.ndarray [(BS x 1)], for each element of the batch assigns a class [0,...,C-1]

		Returns:

			triplet loss (torch.Tensor(), batch-averaged)
pr
		"""
		
		
		#Sample triplets to use for training.
		if self.metric_method=='rM' or self.metric_method =='maha':
			cov = torch.mm(self.L.t(),self.L)
		else:
			cov = torch.tensor(0).cuda()
	
		print('cov',torch.mean(cov),torch.var(cov))

		sampled_triplets = self.sampler.give(batch, labels)
		#Compute triplet loss
		loss = torch.stack([self.triplet_distance(batch[triplet[0],:],batch[triplet[1],:],batch[triplet[2],:],cov) for triplet in sampled_triplets])
		idx = loss>=0
		loss = loss[idx]
		
		return torch.mean(loss)+self.lambda_*torch.sum(torch.abs(torch.sum(batch,1)))/batch.size(0)
