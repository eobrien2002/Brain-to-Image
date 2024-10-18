class Config():
    def __init__(self):
        self.lr = 2.5e-4
        self.min_lr = 0.
        self.weight_decay = 0.05
        self.num_epoch = 300
        self.warmup_epochs = 40
        self.batch_size = 36
        self.clip_grad = 0.8
        
        # Model Parameters
        self.mask_ratio = 0.75
        self.patch_size = 4
        self.embed_dim = 1024 
        self.decoder_embed_dim = 512
        self.depth = 24
        self.num_heads = 16
        self.decoder_num_heads = 16
        self.mlp_ratio = 1.0

        #Other
        self.save_dir = 'checkpoints'
        self.plot_dir = 'plots'
        self.seed = 42
        self.accum_iter = 1
        self.log_steps = 100
        self.data_len = 512
        self.par_number =  [2,3,4,5,6]
#,7,8]
#,9,10]


# To deal with the large data size, we will use the following splits
# iter 1 was AlexMI, BNCI2014002, Cho2017, lee2019
# iter 2 was THINGS particpant 4 and 5
# iter 3 is Ofner and THINGS 6
# iter 4 is THINGS particpant 2, 3, VC data

        self.path_list = [
#     'data/moabb/MunichMI.pickle', #12,000, 2.9gb
    'data/moabb/AlexMI.pickle',  # 8,000, 0.26gb, iter1
   'data/moabb/BNCI2014002.pickle', #48,000, 1.51gb, iter1
#    'data/moabb/BNCI2015004.pickle', #96,000, 2.9gb, 
     'data/moabb/Cho2017.pickle', #5,000 , 0.76 gb, iter1
    'data/moabb/Lee2019_MI.pickle', #5,000, 1.47gb, iter1
   # 'data/moabb/Ofner2017.pickle', #13,000, 2.4gb
 #   'data/moabb/DemonsP300.pickle' #60,000, 1.2gb
]


