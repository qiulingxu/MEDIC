class thing():
    def __init__(self):
        pass
args = thing()
args.nb_iter = 2000
args.print_every = 500
args.anp_eps = 0.2
args.anp_steps = 1
args.anp_alpha = 0.2
args.lr = 0.1
threshold = 0.05
device = 'cuda'