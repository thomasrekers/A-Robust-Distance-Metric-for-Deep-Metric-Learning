##-----------here import the module--------------
import numpy as np
import itertools as it
import torch
#....
#....
#--------------------------------------------

def select_loss(loss,metric,lr,wd,momentum,e_size,num_cls,to_optim):

    """
    Args:
        loss:    name of the loss
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

        criterion = TripletLoss(metric=metric,sampling_method='dist',e_size = e_size) #***
        
        if metric == 'maha' or metric == 'rM':
            to_optim    += [{'params':criterion.parameters(), 'lr': lr,'momentum':momentum, 'weight_decay':wd}]
        
    elif loss == 'contras':
        criterion = ContrasLoss(metric=metric,sampling_method='dist',e_size=e_size)
        
        if metric == 'maha' or metric == 'rM':
            to_optim    += [{'params':criterion.parameters(), 'lr': lr,'momentum':momentum, 'weight_decay':wd}]

    elif loss=='margin':
        criterion = MarginLoss(metric=metric,ssampling_method='dist',e_size=e_size,n_classes=num_cls)
        
        if metric == 'maha' or metric == 'rM':
            to_optim    += [{'params':criterion.parameters(), 'lr': lr,'momentum':momentum, 'weight_decay':wd}]

    elif loss=='lifted':

        criterion    = LiftedLoss(metric = metric,sampling_method='lifted',e_size=e_size)
        
        if metric == 'maha' or metric == 'rM':
            to_optim    += [{'params':criterion.parameters(), 'lr': lr,'momentum':momentum, 'weight_decay':wd}]

    else:
        raise Exception('Loss {} not available!'.format(loss))



    return criterion, to_optim
  
  
  
########################################################################################
## ---------------sampler_class_________________________________________________________
class Sampler():
    
 
    
    def __init__(self,method):
        
        if method == 'rand':
            self.give = self.rand_sampler()
        elif method == 'semi':
            self.give = self.semi_sampler()
        elif method == 'dist':
            self.give = self.dist_sampler()

        elif method == 'lifted':
            self.give = self.lifted_sampler()
        else:
            raise Exception('Sampler {} not available!'.format(method))
    
    
    
    def semi_sampler(self, inputs, labels):

        """
        semihard sampling introduced in 'Deep Metric Learning via Lifted Structured Feature Embedding'.
        Args:
            batch:  np.ndarray or torch.Tensor, batch-wise embedded training samples.
            labels: np.ndarray or torch.Tensor, ground truth labels corresponding to batch.

        Returns:
            bs*(1 anchor ,1 neg,1 pos )
            list of sampled data tuples containing reference indices to the position IN THE BATCH.

        """

        if isinstance(labels, torch.Tensor): labels = labels.detach().numpy()

        bs = inputs.size(0)# batch_size

        #Return distance matrix for all elements in batch (BSxBS)

        distances = self.pdist(inputs.detach()).detach().cpu().numpy()



        positives, negatives = [], []

        anchors = []

        for i in range(bs):

            l, d = labels[i], distances[i]  #  l:label   d: 1*bs [],anchors 与其它sample的距离

            anchors.append(i)

            #1 for batchelements with label l

            neg = labels!=l; pos = labels==l  # Boolean list，1*m  
            # for current anchor
            pos[i] = False   # 去掉自己

            #Find negatives that violate triplet constraint semi-negatives

            neg_mask = np.logical_and(neg,d<d[np.where(pos)[0]].max())  #找出neg_sample 里 比真实相似例子的相似度还高的
            #neg,d< right
            # right = d[np.where(pos)[0]] = d[] , np.where(pos)[0]
            # .max()

            #Find positives that violate triplet constraint semi-hardly

            pos_mask = np.logical_and(pos,d>d[np.where(neg)[0]].min())    # 找出相似度落在 不同例子的最近相似的之外的



            if pos_mask.sum()>0: # if exist

                positives.append(np.random.choice(np.where(pos_mask)[0]))   # boolen 2 indices 如果有，就从里面挑一个

            else:

                positives.append(np.random.choice(np.where(pos)[0]))  # 如果没有找到符合要求的，就随机挑一个



            if neg_mask.sum()>0:

                negatives.append(np.random.choice(np.where(neg_mask)[0]))

            else:

                negatives.append(np.random.choice(np.where(neg)[0]))



        sampled_triplets = [[a, p, n] for a, p, n in zip(anchors, positives, negatives)]   

        return sampled_triplets
    
    
    
    def dist_sampler(self, batch, labels, lower_cutoff=0.5, upper_cutoff=1.4):

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

        if isinstance(labels, torch.Tensor): labels = labels.detach().cpu().numpy()

        bs = batch.shape[0]
        distances    = self.pdist(batch.detach()).clamp(min=lower_cutoff)

        positives, negatives = [],[]
        labels_visited = []
        anchors = []

        for i in range(bs):
            neg = labels!=labels[i]; 
            pos = labels==labels[i]
            q_d_inv = self.inverse_sphere_distances(batch, distances[i], labels, labels[i])
            #Sample positives randomly
            pos[i] = 0 # 去掉自己
            positives.append(np.random.choice(np.where(pos)[0]))   # one sample is returned，
            
            #Sample negatives by distance  
            negatives.append(np.random.choice(bs,p=q_d_inv))  # 0~bs  
            # 存疑https://docs.scipy.org/doc/numpy-1.13.0/reference/generated/numpy.random.choice.html
            

        sampled_triplets = [[a,p,n] for a,p,n in zip(list(range(bs)), positives, negatives)]

        return sampled_triplets
    
    
    def lifted_sampler(self, inputs, labels):

        """
        semihard sampling introduced in 'Deep Metric Learning via Lifted Structured Feature Embedding'.
        Args:
            batch:  np.ndarray or torch.Tensor, batch-wise embedded training samples.
            labels: np.ndarray or torch.Tensor, ground truth labels corresponding to batch.

        Returns:
            bs*(1 anchor ,1 neg,all pos )
            list of sampled data tuples containing reference indices to the position IN THE BATCH.

        """

        if isinstance(labels, torch.Tensor): labels = labels.detach().numpy()

        bs = inputs.size(0)# batch_size

        #Return distance matrix for all elements in batch (BSxBS)

        distances = self.pdist(inputs.detach()).detach().cpu().numpy()



        positives, negatives = [], []

        anchors = []

        for i in range(bs):

            l, d = labels[i], distances[i]  #  l:label   d: 1*bs [],anchors 与其它sample的距离

            anchors.append(i)

            #1 for batchelements with label l

            neg = labels!=l; pos = labels==l  # Boolean list，1*m  
            # for current anchor
            pos[i] = False   # 去掉自己

            #Find negatives that violate triplet constraint semi-negatives

            neg_mask = np.logical_and(neg,d<d[np.where(pos)[0]].max())  #找出neg_sample 里 比真实相似例子的相似度还高的
            #neg,d< right
            # right = d[np.where(pos)[0]] = d[] , np.where(pos)[0]
            # .max()

            #Find positives that violate triplet constraint semi-hardly

            pos_mask = np.logical_and(pos,d>d[np.where(neg)[0]].min())    # 找出相似度落在 不同例子的最近相似的之外的



            if pos_mask.sum()>0: # if exist

                positives.append(np.random.choice(np.where(pos_mask)[0]))   # boolen 2 indices 如果有，就从里面挑一个

            else:

                positives.append(np.random.choice(np.where(pos)[0]))  # 如果没有找到符合要求的，就随机挑一个



          #  if neg_mask.sum()>0:

         #       negatives.append(np.random.choice(np.where(neg_mask)[0],neg_mask.sum() ))   # 去除mask中所有neg

        #    else:

            negatives.append(np.random.choice(np.where(neg)[0],len(np.where(neg_mask)[0]))) # 取出所有的neg


        sampled_lifted = [[a,p,*neg] for a,p,neg in zip(anchors,positives , negatives)]
      

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

        log_q_d_inv[np.where(labels==anchor_label)[0]] = 0



        q_d_inv     = torch.exp(log_q_d_inv - torch.max(log_q_d_inv)) # - max(log) for stability

        #Set sampling probabilities of positives to zero

        q_d_inv[np.where(labels==anchor_label)[0]] = 0



        ### NOTE: Cutting of values with high distances made the results slightly worse.

        # q_d_inv[np.where(dist>upper_cutoff)[0]]    = 0



        #Normalize inverted distance for probability distr.

        q_d_inv = q_d_inv/q_d_inv.sum()

        return q_d_inv.detach().cpu().numpy()


############################################################################
##########------Metric____________________________________________________

class Metric():
    
    '''
    
    '''
    
    def __init__(self,method):
        
        if method == 'E':
            self.dist = self.e_()
        elif method == 'snr':
            self.dist = self.snr_()
        elif method == 'rE':
            self.dist = self.re_()
        elif method = 'maha':
            self.dist = self.m_()
        elif method = 'rM':
            self.dist = self.rm_()
        else:
            raise Exception('Metric {} not available!'.format(method))
    
    def e_(anchor,b,L):
        """
        Return: 
            d
        """
        d = torch.norm(anchor-b)#.pow(2).sum()
        return d
    
    def snr_(anchor,b,L):
        '''
        Args:
            anchor: t
            b: pos or neg
            
        returns:
            d_s
        '''
        tmp =anchor-b
        upper = var(tmp)
        lower = var(anchor)
        d  = upper/lower
        return d
    
    def re_(anchor,b,L):
        '''
        
        return:
            d_s = d_E^2(hi,hj) / d_E^2(hi)
        '''
        
        upper = (anchor-b).pow(2).sum()
        lower = torch.norm(anchor)
        lower = lower.pow(2)
        
        d = upper/lower
        return d
    
    def var(h):
        mean = h.mean()
        n = h.size(0)
        v = (h-mean).pow(2).sum()
        return v/n
    
    def m_(anchor,b,L):
        '''
        L  = torch.ones(m)* sqrt(1/m)
        http://jmlr.csail.mit.edu/papers/volume10/weinberger09a/weinberger09a.pdf
        Args:
            anchor,b (1*m)
            
            ?? size of the output
        
        '''
        #anchor.view(-1,m)
        M = torch.matmul(L.t(),L)
        tmp = anchor -b
        d = torch.matmul(tmp,torch.matmul(M, tmp.t()))
        return d
    
    def rm_(anchor,b,L):
        upper,_,M = m_(anchor,b,L)
        M = torch.matmul(L.t(),L)
        lower = torch.matmul(anchor,torch.matmul(M,anchor.t()))
        d = upper / lower
        return d.pow(2)
        
################################################################################33333
##########3------------------Loss_class-----------------------------------------------


############# Lifted_class
class LiftedLoss(torch.nn.Module):

    def __init__(self, e_size ,margin=1, sampling_method='npair',metric = 'E'):

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
        
        if metric == 'maha' or metric =='rM':
            tmp = 1/e_size
            self.L = torch.nn.Parameter(torch.ones(e_size)/e_size) # initialization
        else:
            self.L = None


    def forward(self, batch, labels):

        """
        Args:
            batch:   torch.Tensor() [(BS x embed_dim)], batch of embeddings
            labels:  np.ndarray [(BS x 1)], for each element of the batch assigns a class [0,...,C-1]
        Returns:
            triplet loss (torch.Tensor(), batch-averaged)

        """

        #Sample triplets to use for training.

        sampled_lifted = self.sampler.give(batch, labels)  # bs* 2+N

        #Compute triplet loss
        
        d_list = []
        for Npair in sample_lifted:
            d_an = 0
            d_pn = 0
            anchor = batch(Npair[0],:)
            pos = batch(Npair[1],:)
            
            d_ap = self.dist.dist(anchor,pos,self.L)
            for neg in Npair[3:]:  
                neg = batch(neg,:)
                d_an += torch.exp(self.margin-self.dist.dist(anchor,neg,self.L))
                d_pn += torch.exp(self.margin-self.dist.dist(pos,neg,self.L))
            
            d = torch.nn.functional.relu(d_ap+torch.log(d_an+d_pn))
            d_list.append(d)
            
        loss =torch.stack(d_list)
        
        return torch.mean(loss)*0.5

##################Margin_loss_cls
class MarginLoss(torch.nn.Module):

    def __init__(self, e_size,n_classes,metric = 'E',margin=0.2, nu=0, beta=1.2, beta_constant=False, sampling_method='dist'):

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

        self.n_classes          = n_classes

        self.beta_constant     = beta_constant



        self.beta_val = beta

        self.beta     = beta if beta_constant else torch.nn.Parameter(torch.ones(n_classes)*beta)

    

        self.nu                 = nu

        self.sampler   = Sampler(method=sampling_method)
        self.dist = Metric(method = metric)
        
        if metric == 'maha' or metric =='rM':
            tmp = 1/e_size
            self.L = torch.nn.Parameter(torch.ones(e_size)/e_size) # initialization
        else:
            self.L = None


    def forward(self, batch, labels):

        """

        Args:

            batch:   torch.Tensor() [(BS x embed_dim)], batch of embeddings

            labels:  np.ndarray [(BS x 1)], for each element of the batch assigns a class [0,...,C-1]

        Returns:

            margin loss (torch.Tensor(), batch-averaged)

        """

        if isinstance(labels, torch.Tensor): labels = labels.detach().cpu().numpy()
        sampled_triplets = self.sampler.give(batch, labels)
        #Compute distances between anchor-positive and anchor-negative.
        d_ap, d_an = [],[]

        for triplet in sampled_triplets:

            train_triplet = {'Anchor': batch[triplet[0],:], 'Positive':batch[triplet[1],:], 'Negative':batch[triplet[2]]}

            pos_dist = self.dist.dist(train_triplet['Anchor'],train_triplet['Positive'],self.L)   

            neg_dist = self.dist.dist(train_triplet['Anchor'],train_triplet['Negative'],self.L)

            d_ap.append(pos_dist)

            d_an.append(neg_dist)

        d_ap, d_an = torch.stack(d_ap), torch.stack(d_an)



        #Group betas together by anchor class in sampled triplets (as each beta belongs to one class).

        if self.beta_constant:

            beta = self.beta

        else:

            beta = torch.stack([self.beta[labels[triplet[0]]] for triplet in sampled_triplets]).type(torch.cuda.FloatTensor)



        #Compute actual margin postive and margin negative loss

        pos_loss = torch.nn.functional.relu(d_ap-beta+self.margin)

        neg_loss = torch.nn.functional.relu(beta-d_an+self.margin)



        #Compute normalization constant

        pair_count = torch.sum((pos_loss>0.)+(neg_loss>0.)).type(torch.cuda.FloatTensor)



        #Actual Margin Loss

        loss = torch.sum(pos_loss+neg_loss) if pair_count==0. else torch.sum(pos_loss+neg_loss)/pair_count



        #(Optional) Add regularization penalty on betas.

        if self.nu: loss = loss + beta_regularisation_loss.type(torch.cuda.FloatTensor)



        return loss
        

############################################################################
###################=----------Triple_loss_cls________________________-
class TripletLoss(torch.nn.Module):

    def __init__(self, e_size,metric = 'E', margin=1, sampling_method='random'):

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
        self.dist = Metric(method = metric)
        
        if metric == 'maha' or metric =='rM':
            tmp = 1/e_size
            self.L = torch.nn.Parameter(torch.ones(e_size)/e_size) # initialization
        else:
            self.L = None


    def triplet_distance(self, anchor, positive, negative):

        """
        Compute triplet loss.
        Args:

            anchor, positive, negative: torch.Tensor(), resp. embeddings for anchor, positive and negative samples.

        Returns:
            triplet loss (torch.Tensor())

        """
        d_ap = self.dist.dist(anchor,positive,self.L)
        d_an = self.dist.dist(anchor,negative,self.L)
        return torch.nn.functional.relu( d_ap - d_an + self.margin )
    
    def forward(self, batch, labels):

        """
        Args:

            batch:   torch.Tensor() [(BS x embed_dim)], batch of embeddings

            labels:  np.ndarray [(BS x 1)], for each element of the batch assigns a class [0,...,C-1]

        Returns:

            triplet loss (torch.Tensor(), batch-averaged)

        """

        #Sample triplets to use for training.

        sampled_triplets = self.sampler.give(batch, labels)

        #Compute triplet loss

        loss = torch.stack([self.triplet_distance(batch[triplet[0],:],batch[triplet[1],:],batch[triplet[2],:]) for triplet in sampled_triplets])

        return torch.mean(loss)

        
        
  
  
  
  
  
  
  
  
  
  
  
  
  
  
 
