import argparse

def get_arguments():
    parser = argparse.ArgumentParser()

    # various path
    parser.add_argument('--checkpoint_root', type=str, default='./weight/erasing_net', help='models weight are saved here')
    parser.add_argument('--log_root', type=str, default='./results', help='logs are saved here')
    parser.add_argument('--dataset', type=str, default='CIFAR10', help='name of image dataset')
    parser.add_argument('--trojai-model-id',type=int,default=6)
    parser.add_argument('--s_model', type=str, default='./weight/s_net/WRN-16-1-S-model_best.pth.tar', help='path of student model')
    parser.add_argument('--t_model', type=str, default='./weight/t_net/WRN-16-1-T-model_best.pth.tar', help='path of teacher model')

    # training hyper parameters
    parser.add_argument('--print_freq', type=int, default=10, help='frequency of showing training results on console')
    parser.add_argument('--epochs', type=int, default=10, help='number of total epochs to run')
    parser.add_argument('--batch_size', type=int, default=64, help='The size of batch')
    parser.add_argument('--lr', type=float, default=0.1, help='initial learning rate')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='weight decay')
    parser.add_argument('--num_class', type=int, default=10, help='number of classes')
    parser.add_argument('--ratio', type=float, default=0.05, help='ratio of training data')


    parser.add_argument('--threshold_clean', type=float, default=70.0, help='threshold of save weight')
    parser.add_argument('--threshold_bad', type=float, default=90.0, help='threshold of save weight')
    parser.add_argument('--cuda', type=int, default=1)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--save', type=int, default=1)

    # others
    parser.add_argument('--seed', type=int, default=2, help='random seed')
    parser.add_argument('--note', type=str, default='try', help='note for this run')
    parser.add_argument("--converge",action="store_true", help='Apply cosine annealing to learning rate.')

    # net and dataset choosen
    parser.add_argument('--data_name', type=str, default='CIFAR10', help='name of dataset')
    parser.add_argument('--t_name', type=str, default='WRN-16-1', help='name of teacher')
    parser.add_argument('--s_name', type=str, default='WRN-16-1', help='name of student')
    parser.add_argument("--clone_start",type=int,default=0, help = 'starting epoch for cloning')
    parser.add_argument("--clone_end",type=int,default=100, help='ending epoch for cloning')


    # MEDIC parameter
    parser.add_argument("--hook", action="store_true",
                        help="Enable the MEDIC function, it basically adds hooks to internal layers' output for loss calculation.")
    parser.add_argument("--hookweight",type=float,default=10, help="Weight of cloning loss. It is the lambda in the paper.")
    parser.add_argument("--hook-plane", type=str, default="conv",
                        choices=["conv","bn","conv+bn", "conv+bn+dense"],
                        help="Which layers to add cloning loss to.")
    parser.add_argument("--imp_temp", type=float, default=1.0,
                        help="Temperature of the softmax function in the MEDIC loss.")
    parser.add_argument("--keepstat", action = "store_true",
                        help='Maintaining the statistics for intermediate layers. This is required for estimating the mean and variance of intermediate layers.')
    parser.add_argument("--norml2", action = "store_true",
                        help = 'Normalize the neurons outputs by mean and standard deviation')
    parser.add_argument("--isample",type=str,default="", choices=["", "l2",],
                        help="Which importance sampling method to use for MEDIC loss.")
    parser.add_argument("--sbn", type=str, default="batch",
                        help ='Which estimation of statistics to use for estimating the mean and variance of intermediate layers from student model. Batch means running estimation. exact means using per-batch estimation.')
    parser.add_argument("--tbn", type=str, default="exact",
                        help ='Which estimation of statistics to use for estimating the mean and variance of intermediate layers from teacher model. Batch means running estimation. exact means using per-batch estimation.')

    

    ## Different removal methods, choices are likely exclusive
    parser.add_argument("--MCR",action="store_true")
    parser.add_argument("--lwf", action="store_true", help="Apply knowledge distillation to clean the backdoor model.")
    parser.add_argument("--scratch", action="store_true", help="Train from scratch to remove any backdoor model.")    
    parser.add_argument("--pretrain", action="store_true", help='Starts from backdoor teacher model.')
    parser.add_argument("--Finetune", action="store_true", help='Apply finetuning to clean the backdoor model.')

    # NAD parameter
    parser.add_argument("--NAD",action="store_true", help='Apply NAD to clean the backdoor model.')
    parser.add_argument('--beta1', type=int, default=0, help='beta of low layer')
    parser.add_argument('--beta2', type=int, default=0, help='beta of middle layer')
    parser.add_argument('--beta3', type=int, default=5000, help='beta of high layer')
    parser.add_argument('--p', type=float, default=2.0, help='power for AT')
    

    parser.add_argument("--fineprune",action="store_true", help='Apply fineprune to clean the backdoor model.')
    parser.add_argument("--prune-num",type=int, default=32, help='Number of filters to prune in fineprune.')
    parser.add_argument("--ANP", action = "store_true", help='Apply ANP to clean the backdoor model.')


    parser.add_argument("--log_name", default = "dev")    
    parser.add_argument("--case",default = "")

    # Used for exhibiting the strong and fragile backdoor effects.
    parser.add_argument("--disaug",action="store_true", help='disable applying augmentations')

    # backdoor attacks
    parser.add_argument('--inject_portion', type=float, default=0.1, help='ratio of backdoor samples')
    parser.add_argument('--target_label', type=int, default=5, help='class of target label')
    parser.add_argument('--trigger_type', type=str, default='gridTrigger', help='type of backdoor trigger')
    parser.add_argument('--target_type', type=str, default='all2one', help='type of backdoor label')
    parser.add_argument('--trig_w', type=int, default=3, help='width of trigger pattern')
    parser.add_argument('--trig_h', type=int, default=3, help='height of trigger pattern')
    parser.add_argument("--clean_label", action="store_true")

    return parser


def name(opt):
    assert (opt.Finetune ^ opt.NAD ^ opt.hook ^ opt.fineprune ^ opt.MCR ^ opt.ANP ^ opt.scratch) or opt.lwf
    if opt.hook:
        if opt.keepstat:
            opt1 = "STs{}_t{}".format(opt.sbn,opt.tbn)
        else:
            opt1 = ""
        if opt.lwf:
            opt1 += "_LWF"
        Method = "#Mhook_P{}_Ls{}_T{:.2e}_La{:.2e}_L2N{}_{}".format(opt.hook_plane,opt.isample,opt.imp_temp,opt.hookweight, opt.norml2, opt1)
    elif opt.NAD:
        Method = "#Mnad_b{:.2e}{:.2e}{:.2e}".format(opt.beta1,opt.beta2,opt.beta3)
    elif opt.Finetune:
        Method = "#Mfinetune"
    elif opt.fineprune:
        Method = "#Fineprune{:d}".format(opt.prune_num)
    elif opt.scratch:
        Method = "#Scratch"
    elif opt.MCR:
        Method = "#MCR0.1"
    elif opt.ANP:
        Method = "#ANP0.2_EPS0.2_LR0.1"
    elif opt.lwf:
        Method = "#lwf"
    else:
        assert False
    Conf = "#Ep{}_cs{}_ce{}_cv{}".format(opt.epochs, opt.clone_start, opt.clone_end, opt.converge)
    if opt.ratio != 0.05 and opt.ratio != 0.005:
        Conf+= "RT{:2e}".format(opt.ratio)
    name = "#C{}#T{}{}{}".format(opt.case,opt.trial_id,Method, Conf)
    
    return name

opt = get_arguments().parse_args()