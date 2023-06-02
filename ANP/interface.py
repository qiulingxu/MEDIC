from .optimize_mask_cifar import *
from .prune_neuron_cifar import *
from .config import *
import random
import string
def rand_str():
    letters = string.ascii_lowercase
    return ''.join(random.choice(letters) for i in range(10)) 



def prune(net, trainloader):


    net = net.to(device)
    criterion = torch.nn.CrossEntropyLoss().to(device)

    parameters = list(net.named_parameters())
    #print([i0 for i0,i1 in parameters])
    mask_params = [v for n, v in parameters if "neuron_mask" in n]
    mask_optimizer = torch.optim.SGD(mask_params, lr=args.lr, momentum=0.9)
    
    noise_params = [v for n, v in parameters if "neuron_noise" in n]
    noise_optimizer = torch.optim.SGD(noise_params, lr=args.anp_eps / args.anp_steps)
    #print(mask_params, noise_params)

    print('Iter \t lr \t Time \t TrainLoss \t TrainACC \t PoisonLoss \t PoisonACC \t CleanLoss \t CleanACC')
    nb_repeat = int(np.ceil(args.nb_iter / args.print_every))
    for i in range(nb_repeat):
        start = time.time()
        lr = mask_optimizer.param_groups[0]['lr']
        train_loss, train_acc = mask_train(model=net, criterion=criterion, data_loader=trainloader,
                                           mask_opt=mask_optimizer, noise_opt=noise_optimizer)
        end = time.time()
        print('{} \t {:.3f} \t {:.1f} \t {:.4f} \t {:.4f}'.format(
            (i + 1) * args.print_every, lr, end - start, train_loss, train_acc))
    fn = rand_str() + ".txt"
    mask_file = os.path.join("ANP", fn)
    save_mask_scores(net.state_dict(), mask_file)    

    mask_values = read_data(mask_file)
    mask_values = sorted(mask_values, key=lambda x: float(x[2]))
    start = 0
    for idx in range(start, len(mask_values)):
        if float(mask_values[idx][2]) <= threshold:
            pruning(net, mask_values[idx])
            start += 1
        else:
            break
    net.eval()
    exclude_noise(net)
    return net