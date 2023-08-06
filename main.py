from torch import nn
import torch as T
from torch.nn import functional as F
from models.selector import *
from utils.util import *
from data_loader import get_train_loader, get_test_loader, set_trojai_model_id
from at import AT
import config
import sys
import copy
import json
from config import get_arguments
from hook import Hook, hookwisemse, hookwisemse_imp_sample, hookwisemse_at
from timeit import default_timer as timer
from prune import prune
from functools import partial
from tqdm import tqdm

def loss_fn_kd(outputs, labels, teacher_outputs, params):
    """
    Compute the knowledge-distillation (KD) loss given outputs, labels.
    "Hyperparameters": temperature and alpha
    NOTE: the KL Divergence for PyTorch comparing the softmaxs of teacher
    and student expects the input tensor to be log probabilities! See Issue #2
    """
    alpha = 1.0
    T = 2
    KD_loss = nn.KLDivLoss()(F.log_softmax(outputs/T, dim=1),
                             F.softmax(teacher_outputs/T, dim=1)) * (alpha * T * T)

    return KD_loss



def train_step(opt, train_loader, nets, optimizer, criterions, epoch):
    at_losses = AverageMeter()
    hk_losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    snet = nets['snet']
    tnet = nets['tnet']

    snet.train()
    if opt.nobndiv:
        for m in tnet.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.train()
                m.track_running_stats = False
    #for m1, m2 in zip(snet.modules(), tnet.modules()):
    #    if isinstance(m1, nn.BatchNorm2d):
    #        m1.running_mean = T.clone(m2.running_mean) * rand (m2.running_mean,0.9, 1.1)
    #        m1.running_var = T.clone(m2.running_var) * rand (m2.running_var,0.9, 1.1)
    #for m in snet.modules():
    #    if isinstance(m, nn.BatchNorm2d):
    #        m.eval()
    #        m.track_running_stats = False            
    criterionCls = criterions['criterionCls']
    criterionAT = criterions['criterionAT']

    
    for idx, (img, target) in tqdm(enumerate(train_loader, start=1)):
        if opt.cuda:
            img = img.cuda()
            target = target.cuda()
            img.requires_grad=True
        if opt.hook:
            with shook as h2:
                output_s = snet(img)
            with thook as h1:
                output_t = tnet(img)
        else:
            output_s = snet(img)
            output_t = tnet(img)
        at_loss = T.tensor(0.).cuda()
        if opt.lwf:        
            at_loss += loss_fn_kd(output_s, target, output_t, None) * opt.hookweight
        cls_loss = criterionCls(output_s, target)
        if opt.hook and epoch>=opt.clone_start and epoch<=opt.clone_end:
            if opt.isample == "l2":
                _cls_loss = criterionCls(output_t, target) + cls_loss
                hk_loss = hookwisemse_imp_sample(h1, h2, _cls_loss)
            elif opt.isample == "":
                hk_loss = hookwisemse(h1, h2)
            elif opt.isample == "at":
                hk_loss = hookwisemse_at(h1, h2)
            else:
                assert False
            at_loss += hk_loss * config.opt.hookweight
        else:
            hk_loss = T.zeros((1,))
        # For comparing NAD
        if float(opt.beta3)!= 0.:
            at3_loss = criterionAT(snet.activation3, tnet.activation3) * opt.beta3
        else:
            at3_loss = 0.
        if float(opt.beta2)!=0.:
            at2_loss = criterionAT(snet.activation2, tnet.activation2) * opt.beta2
        else:
            at2_loss = 0.
        if float(opt.beta1)!=0.:
            at1_loss = criterionAT(snet.activation1, tnet.activation1) * opt.beta1
        else:
            at1_loss = 0.
        at_loss += at1_loss + at2_loss + at3_loss + cls_loss

        prec1, prec5 = accuracy(output_s, target, topk=(1, 5))
        at_losses.update(at_loss.item(), img.size(0))
        top1.update(prec1.item(), img.size(0))
        top5.update(prec5.item(), img.size(0))
        if opt.hook:
            hk_losses.update(hk_loss.item(), img.size(0))

        optimizer.zero_grad()
        at_loss.backward()
        if opt.rw:
            for param in snet.parameters():
                if param.requires_grad:
                    rv =  T.normal(T.zeros_like(param), T.ones_like(param))
                    nm = T.sqrt(T.sum(T.square(rv)))
                    nm_p = T.sqrt(T.sum(T.square(param.grad)))
                    lr = optimizer.param_groups[0]["lr"]
                    rv = rv / nm *0.1
                    parrallel = T.dot(T.flatten(rv), T.flatten(param.grad)) *param.grad
                    rv = rv * nm_p
                    indp = rv - parrallel 
                    param.grad.data = indp.detach()
        optimizer.step()
        if idx % opt.print_freq == 0:
            print('Epoch[{0}]:[{1:03}/{2:03}] '
                  'AT_loss:{losses.val:.4f}({losses.avg:.4f})  '
                  'HK_loss:{hk_losses.val:.4f}({hk_losses.avg:.4f})  '
                  'prec@1:{top1.val:.2f}({top1.avg:.2f})  '
                  'prec@5:{top5.val:.2f}({top5.avg:.2f})'.format(epoch, idx, len(train_loader),hk_losses=hk_losses, losses=at_losses, top1=top1, top5=top5))


def test(opt, test_clean_loader, test_bad_loader, nets, criterions, epoch):
    test_process = []
    top1 = AverageMeter()
    top5 = AverageMeter()

    snet = nets['snet']
    #tnet = nets['tnet']

    criterionCls = criterions['criterionCls']

    snet.eval()
    for idx, (img, target) in enumerate(test_clean_loader, start=1):
        img = img.cuda()
        target = target.cuda()
        with torch.no_grad():
            output_s = snet(img)

        prec1, prec5 = accuracy(output_s, target, topk=(1, 5))
        top1.update(prec1.item(), img.size(0))
        top5.update(prec5.item(), img.size(0))

    acc_clean = [top1.avg, top5.avg]

    cls_losses = AverageMeter()
    at_losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    for idx, (img, target) in enumerate(test_bad_loader, start=1):
        img = img.cuda()
        img.requires_grad=True
        target = target.cuda()
        at_loss = T.tensor(0.).cuda()
        with torch.no_grad():
            output_s = snet(img)
        cls_loss = criterionCls(output_s, target)
        at_loss = 0.
        

        prec1, prec5 = accuracy(output_s, target, topk=(1, 5))
        cls_losses.update(cls_loss.item(), img.size(0))
        at_losses.update(at_loss, img.size(0)) #.item()
        top1.update(prec1.item(), img.size(0))
        top5.update(prec5.item(), img.size(0))

    acc_bd = [top1.avg, top5.avg, cls_losses.avg, at_losses.avg]

    print('[clean]Prec@1: {:.2f}'.format(acc_clean[0]))
    print('[bad]Prec@1: {:.2f}'.format(acc_bd[0]))

    # save training progress
    log_root = opt.log_root + '/results.csv'
    test_process.append(
        (epoch, acc_clean[0], acc_bd[0], acc_bd[2], acc_bd[3]))
    df = pd.DataFrame(test_process, columns=(
    "epoch", "test_clean_acc", "test_bad_acc", "test_bad_cls_loss", "test_bad_at_loss"))
    df.to_csv(log_root, mode='a', index=False, encoding='utf-8')

    return acc_clean, acc_bd

def adjust_learning_rate_medic(optimizer, epoch, lr):
    # Cloning requires a bit larger learning rate than normal finetuneing
    if epoch < 40:
        lr = 0.01
    elif epoch < 70:
        lr = 0.001
    elif epoch < 100:
        lr = 0.0001
    else:
        lr = 0.0001
    print('epoch: {}  lr: {:.4f}'.format(epoch, lr))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def train(opt, teacher = None, student = None):
    # Load models
    ret = {}
    print('----------- Network Initialization --------------')
    if teacher is None:
        teacher = select_model(dataset=opt.data_name,
                            model_name=opt.t_name,
                            pretrained=True,
                            pretrained_models_path=opt.t_model,
                            n_classes=opt.num_class).to(opt.device)
    print('finished teacher model init...')
    if opt.pretrain:
        pretrained = True
    else:
        if opt.lwf or opt.hook or opt.scratch:
            pretrained = False
        else:
            pretrained = True
    if student is None:
        student = select_model(dataset=opt.data_name,
                            model_name=opt.s_name,
                            pretrained=pretrained,
                            pretrained_models_path=opt.s_model,
                            n_classes=opt.num_class).to(opt.device)
    print('finished student model init...')
    teacher.eval()

    nets = {'snet': student, 'tnet': teacher}

    for param in teacher.parameters():
        param.requires_grad = False

    # initialize optimizer
    if opt.hook:
        optimizer = torch.optim.Adam(student.parameters(),
                                    lr=opt.lr,
                                    weight_decay=opt.weight_decay,
                                    )
    else:
        optimizer = torch.optim.SGD(student.parameters(),
                                    lr=opt.lr,
                                    momentum=opt.momentum,
                                    weight_decay=opt.weight_decay,
                                    nesterov=True)
    if opt.converge:
        schedule=T.optim.lr_scheduler.CosineAnnealingLR(optimizer, opt.epochs)
    # define loss functions
    if opt.cuda:
        criterionCls = nn.CrossEntropyLoss().cuda()
        criterionAT = AT(opt.p)
    else:
        criterionCls = nn.CrossEntropyLoss()
        criterionAT = AT(opt.p)

    print('----------- DATA Initialization --------------')
    train_loader = get_train_loader(opt)
    if opt.fineprune:
        prune(partial(student,prune=True),train_loader,student.setprunemask,leastk=opt.prune_num)
    test_clean_loader, test_bad_loader = get_test_loader(opt)
    if opt.hook and opt.keepstat:
        thook.train()
        for idx, (img, target) in tqdm(enumerate(train_loader, start=1)):
            if opt.cuda:
                img = img.cuda()
                target = target.cuda()
                img.requires_grad=True
            if opt.hook:
                with thook as h1:
                    output_t = teacher(img)
            if idx>100:
                break
        thook.eval()

    print('----------- Train Initialization --------------')
    for epoch in range(0, opt.epochs):
        if not opt.converge:
            if opt.lwf:
                adjust_learning_rate_medic(optimizer, epoch, opt.lr)
            elif opt.hook:
                adjust_learning_rate_medic(optimizer, epoch, opt.lr)
            else:
                adjust_learning_rate(optimizer, epoch, opt.lr)

        # train every epoch
        criterions = {'criterionCls': criterionCls, 'criterionAT': criterionAT}

        if epoch == 0:
            # before training test firstly
            test(opt, test_clean_loader, test_bad_loader, nets,
                                         criterions, epoch)

        train_step(opt, train_loader, nets, optimizer, criterions, epoch+1)

        # evaluate on testing set
        print('testing the models......')
        acc_clean, acc_bad = test(opt, test_clean_loader, test_bad_loader, nets, criterions, epoch+1)
        if opt.converge:
            schedule.step()
        # remember best precision and save checkpoint
        # save_root = opt.checkpoint_root + '/' + opt.s_name
        if opt.save:
            os.makedirs(opt.checkpoint_root,exist_ok=True)
            is_best = acc_clean[0] > opt.threshold_clean
            print("saving", is_best)
            opt.threshold_clean = min(acc_bad[0], opt.threshold_clean)

            best_clean_acc = acc_clean[0]
            best_bad_acc = acc_bad[0]
            ret["best_clean_acc"] = best_clean_acc
            ret["best_bad_acc"] = best_bad_acc

            save_checkpoint({
                'epoch': epoch,
                'state_dict': student.state_dict(),
                'best_clean_acc': best_clean_acc,
                'best_bad_acc': best_bad_acc,
                'optimizer': optimizer.state_dict(),
            }, is_best, opt.checkpoint_root, opt.s_name)
    return student, ret

def finetune(opt):
    ret = {}
    # Load models
    print('----------- Network Initialization --------------')
    teacher = select_model(dataset=opt.data_name,
                           model_name=opt.t_name,
                           pretrained=True,
                           pretrained_models_path=opt.t_model,
                           n_classes=opt.num_class).to(opt.device)
    print('finished teacher model init...')

    student = select_model(dataset=opt.data_name,
                           model_name=opt.s_name,
                           pretrained=True,
                           pretrained_models_path=opt.s_model,
                           n_classes=opt.num_class).to(opt.device)
    print('finished student model init...')
    teacher.eval()

    nets = {'snet': student, 'tnet': teacher}

    for param in teacher.parameters():
        param.requires_grad = False
    
    # initialize optimizer

    optimizer = torch.optim.SGD(student.parameters(),
                                lr=opt.lr,
                                momentum=opt.momentum,
                                weight_decay=opt.weight_decay,
                                nesterov=True)
    # define loss functions
    if opt.cuda:
        criterionCls = nn.CrossEntropyLoss().cuda()
        criterionAT = AT(opt.p)
    else:
        criterionCls = nn.CrossEntropyLoss()
        criterionAT = AT(opt.p)

    print('----------- DATA Initialization --------------')
    train_loader = get_train_loader(opt)
    if opt.fineprune:
        prune(partial(student,prune=True),train_loader,student.setprunemask)
    test_clean_loader, test_bad_loader = get_test_loader(opt)

    print('----------- Train Initialization --------------')
    for epoch in range(0, opt.epochs):

        adjust_learning_rate(optimizer, epoch, opt.lr)

        # train every epoch
        criterions = {'criterionCls': criterionCls, 'criterionAT': criterionAT}

        if epoch == 0:
            # before training test firstly
            test(opt, test_clean_loader, test_bad_loader, nets,
                                         criterions, epoch)

        train_step(opt, train_loader, nets, optimizer, criterions, epoch+1)

        # evaluate on testing set
        print('testing the models......')
        acc_clean, acc_bad = test(opt, test_clean_loader, test_bad_loader, nets, criterions, epoch+1)

        # remember best precision and save checkpoint
        # save_root = opt.checkpoint_root + '/' + opt.s_name
        if opt.save:
            os.makedirs(opt.checkpoint_root,exist_ok=True)
            is_best = acc_clean[0] > opt.threshold_clean
            opt.threshold_clean = min(acc_bad[0], opt.threshold_clean)

            best_clean_acc = acc_clean[0]
            best_bad_acc = acc_bad[0]
            ret["best_clean_acc"] = best_clean_acc
            ret["best_bad_acc"] = best_bad_acc
            save_checkpoint({
                'epoch': epoch,
                'state_dict': student.state_dict(),
                'best_clean_acc': best_clean_acc,
                'best_bad_acc': best_bad_acc,
                'optimizer': optimizer.state_dict(),
            }, is_best, opt.checkpoint_root, opt.s_name)
    return student, ret 

def test_only(opt, model, epoch):
    test_clean_loader, test_bad_loader = get_test_loader(opt)
    criterionCls = nn.CrossEntropyLoss().cuda()
    criterions = {'criterionCls': criterionCls}
    nets = {'snet':model}
    acc_clean, acc_bad = test(opt, test_clean_loader, test_bad_loader, nets, criterions, epoch)
    ret = {}
    ret["best_clean_acc"] = acc_clean
    ret["best_bad_acc"] = acc_bad
    return ret 

def main():
    # Prepare arguments
    opt = get_arguments().parse_args()
    config.opt = opt
    start = timer()
    global shook, thook
    shook = Hook(train=True,keepstat=opt.keepstat, nmode=opt.sbn)
    thook = Hook(keepstat=opt.keepstat, nmode=opt.tbn)
    shook.train()
    thook.eval()


    if opt.NAD:
        opt1 = copy.copy(opt)
        
        opt1.beta1 = 0 
        opt1.beta2 = 0
        opt1.beta3 = 0
        opt1.epochs = 10
        opt1.save = 0
        opt1.checkpoint_root += "_finetune"
        config.opt = opt1
        _name = config.name(opt)
        student, _ = finetune(opt1)
        opt1 = copy.copy(opt)
        opt1.save = 1
        #opt1.epochs = 30
        config.opt = opt1
        _name = config.name(opt1)
        logdir = os.path.join("logs",opt.log_name, _name)
        os.makedirs(logdir,exist_ok=True)
        opt1.log_root = logdir
        opt1.checkpoint_root = os.path.join("weight",opt.log_name, _name)
        _, ret = train(opt1, teacher=student)
    elif opt.Finetune:
        config.opt = opt
        opt.beta1 = 0 
        opt.beta2 = 0
        opt.beta3 = 0
        #opt.epochs = 30
        _name = config.name(opt)
        logdir = os.path.join("logs",opt.log_name, _name)
        os.makedirs(logdir,exist_ok=True)
        opt.log_root = logdir
        opt.checkpoint_root = os.path.join("weight",opt.log_name, _name)
        student, ret = finetune(opt)            
    elif opt.hook:
        _name = config.name(opt)
        logdir = os.path.join("logs",opt.log_name, _name)
        os.makedirs(logdir,exist_ok=True)
        opt.log_root = logdir
        opt.checkpoint_root = os.path.join("weight",opt.log_name, _name)
        _, ret = train(opt)
    elif opt.scratch or opt.lwf:
        _name = config.name(opt)
        logdir = os.path.join("logs", opt.log_name, _name)
        os.makedirs(logdir, exist_ok=True)
        opt.log_root = logdir
        opt.checkpoint_root = os.path.join("weight", opt.log_name, _name)
        _, ret = train(opt)
    elif opt.fineprune:
        config.opt = opt
        opt.beta1 = 0 
        opt.beta2 = 0
        opt.beta3 = 0
        #opt.epochs = 30
        _name = config.name(opt)
        logdir = os.path.join("logs",opt.log_name, _name)
        os.makedirs(logdir,exist_ok=True)
        opt.log_root = logdir
        opt.checkpoint_root = os.path.join("weight",opt.log_name, _name)
        student, ret = finetune(opt)       
    elif opt.MCR:
        config.opt = opt
        opt.beta1 = 0 
        opt.beta2 = 0
        opt.beta3 = 0
        model_tot = select_MCR_model(model_name=opt.t_name,n_classes=opt.num_class)
        
        model = model_tot.net
        #opt.epochs = 30
        _name = config.name(opt)
        logdir = os.path.join("logs",opt.log_name, _name)
        os.makedirs(logdir,exist_ok=True)
        opt.log_root = logdir
        opt.checkpoint_root = os.path.join("weight",opt.log_name, _name)
        student, ret = finetune(opt)            
        teacher = select_model(dataset=opt.data_name,
                        model_name=opt.t_name,
                        pretrained=True,
                        pretrained_models_path=opt.t_model,
                        n_classes=opt.num_class).to(opt.device)
        model.load_state_dict(student.state_dict(),point=0)
        model.load_state_dict(teacher.state_dict(),point=2)
        model_tot.init_linear()
        model_tot.cuda()
        train_loader = get_train_loader(opt)
        from MCR.train import MCR_train
        from MCR.utils import update_bn
        model_tot.fix_t(None)
        model_tot = MCR_train(model_tot,train_loader)
        model_tot.fix_t(0.1)
        model_tot.train()
        update_bn(train_loader, model_tot)
        ret = test_only(opt, model_tot,100)
                     
    elif opt.ANP:
        from ANP.interface import prune
        config.opt = opt
        opt.beta1 = 0 
        opt.beta2 = 0
        opt.beta3 = 0
        teacher = select_model(dataset=opt.data_name,
                        model_name=opt.t_name,
                        pretrained=True,
                        pretrained_models_path=opt.t_model,
                        n_classes=opt.num_class).to(opt.device)        
        train_loader = get_train_loader(opt)
        prune(teacher,train_loader)
        ret = test_only(opt, teacher,100)
    else:
        assert False
    

    end = timer()
    js = {"arg": vars(opt), "ret":ret, "time":end-start}
    with open(os.path.join("logs",opt.log_name,"Exp_log.txt"),"a") as fo:
        fo.write(json.dumps(js) +"\r\n")

if (__name__ == '__main__'):
    
    main()
    
