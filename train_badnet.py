import faulthandler
faulthandler.enable()
from models.selector import *
from utils.util import *
from data_loader import get_test_loader, get_backdoor_loader
from config import get_arguments
import torchattacks
import torch as T

def train_step(opt, train_loader, nets, optimizer, criterions, epoch):
    cls_losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    snet = nets['snet']

    criterionCls = criterions['criterionCls']
    snet.train()

    for idx, (img, target, is_inject) in enumerate(train_loader, start=1):
        if opt.cuda:
            img = img.cuda()
            target = target.cuda()
            #print(img.size())
            if opt.clean_label:
                is_inject = is_inject.cuda()
                #print(is_inject)
                img_adv = atk(img, target)
                tlabel = T.zeros_like(target).fill_(opt.target_label)
                mask = T.logical_and(T.eq(target,tlabel),is_inject).reshape((-1,1,1,1))
                #print(mask)
                img = T.where(mask, img_adv, img)

        output_s = snet(img)

        cls_loss = criterionCls(output_s, target)

        prec1, prec5 = accuracy(output_s, target, topk=(1, 5))
        cls_losses.update(cls_loss.item(), img.size(0))
        top1.update(prec1.item(), img.size(0))
        top5.update(prec5.item(), img.size(0))

        optimizer.zero_grad()
        cls_loss.backward()
        optimizer.step()

        if idx % opt.print_freq == 0:
            print('Epoch[{0}]:[{1:03}/{2:03}] '
                  'cls_loss:{losses.val:.4f}({losses.avg:.4f})  '
                  'prec@1:{top1.val:.2f}({top1.avg:.2f})  '
                  'prec@5:{top5.val:.2f}({top5.avg:.2f})'.format(epoch, idx, len(train_loader), losses=cls_losses, top1=top1, top5=top5))


def test(opt, test_clean_loader, test_bad_loader, nets, criterions, epoch):
    test_process = []
    top1 = AverageMeter()
    top5 = AverageMeter()

    snet = nets['snet']
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
        target = target.cuda()

        with torch.no_grad():
            output_s = snet(img)
            cls_loss = criterionCls(output_s, target)

        prec1, prec5 = accuracy(output_s, target, topk=(1, 5))
        cls_losses.update(cls_loss.item(), img.size(0))
        top1.update(prec1.item(), img.size(0))
        top5.update(prec5.item(), img.size(0))

    acc_bd = [top1.avg, top5.avg, cls_losses.avg]

    print('[clean]Prec@1: {:.2f}'.format(acc_clean[0]))
    print('[bad]Prec@1: {:.2f}'.format(acc_bd[0]))

    # save training progress
    log_root = opt.log_root + '/backdoor_results.csv'
    test_process.append(
        (epoch, acc_clean[0], acc_bd[0], acc_bd[2]))
    df = pd.DataFrame(test_process, columns=(
    "epoch", "test_clean_acc", "test_bad_acc", "test_bad_cls_loss"))
    df.to_csv(log_root, mode='a', index=False, encoding='utf-8')

    return acc_clean, acc_bd


def train(opt):
    # Load models
    print('----------- Network Initialization --------------')
    student = select_model(dataset=opt.data_name,
                           model_name=opt.s_name,
                           pretrained=False,
                           pretrained_models_path=opt.s_model,
                           n_classes=opt.num_class).to(opt.device)
    print('finished student model init...')

    nets = {'snet': student}

    # initialize optimizer
    optimizer = torch.optim.SGD(student.parameters(),
                                lr=opt.lr,
                                momentum=opt.momentum,
                                weight_decay=opt.weight_decay,
                                nesterov=True)

    # define loss functions
    if opt.cuda:
        criterionCls = nn.CrossEntropyLoss().cuda()
    else:
        criterionCls = nn.CrossEntropyLoss()
    
    if opt.clean_label:
        global atk
        #atk = torchattacks.PGDL2(student, eps=300, alpha=10, steps=30)
        if opt.dataset == "Trojai":
            atk = torchattacks.PGD(student, eps=2/255.,
                                   alpha=0.5/255., steps=10)
        else:
            atk = torchattacks.PGD(student, eps=16/255., alpha=2/255., steps=10)

    print('----------- DATA Initialization --------------')
    train_loader = get_backdoor_loader(opt)
    test_clean_loader, test_bad_loader = get_test_loader(opt)
    if opt.converge:
        schedule=T.optim.lr_scheduler.CosineAnnealingLR(optimizer, opt.epochs)
    print('----------- Train Initialization --------------')
    for epoch in range(1, opt.epochs):
        if not opt.converge:
            _adjust_learning_rate(optimizer, epoch, opt.lr)

        # train every epoch
        criterions = {'criterionCls': criterionCls}
        train_step(opt, train_loader, nets, optimizer, criterions, epoch)

        # evaluate on testing set
        print('testing the models......')
        acc_clean, acc_bad = test(opt, test_clean_loader, test_bad_loader, nets, criterions, epoch)

        # remember best precision and save checkpoint
        if opt.save:
            is_best = acc_bad[0] > opt.threshold_bad
            opt.threshold_bad = min(acc_bad[0], opt.threshold_bad)

            best_clean_acc = acc_clean[0]
            best_bad_acc = acc_bad[0]

            s_name = opt.s_name + '-S-model_best.pth'
            save_checkpoint({
                'epoch': epoch,
                'state_dict': student.state_dict(),
                'best_clean_acc': best_clean_acc,
                'best_bad_acc': best_bad_acc,
                'optimizer': optimizer.state_dict(),
            }, is_best, opt.checkpoint_root, s_name)
        if opt.converge:
            schedule.step()            


def _adjust_learning_rate(optimizer, epoch, lr):
    if config.opt.converge:
        if epoch < 40:
            lr = lr
        elif epoch < 70:
            lr = 0.01 * lr
        else:
            lr = 0.0009        
    else:
        if epoch < 21:
            lr = lr
        elif epoch < 30:
            lr = 0.01 * lr
        else:
            lr = 0.0009
        print('epoch: {}  lr: {:.4f}'.format(epoch, lr))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def main():
    # Prepare arguments
    opt = get_arguments().parse_args()
    #if opt.converge:
    #    opt.epochs=100
    config.opt = opt
    train(opt)

if (__name__ == '__main__'):
    main()
