import argparse


def get_parser(method,study,mode):
    parser = argparse.ArgumentParser()
    parser.add_argument('--rho', type=int, default=1,
                        help='\sigma^2/\nu')    
    parser.add_argument('--ep', type=int, default=10,
                        help='number of epochs')  
    parser.add_argument('--trainingSize', type=int, default=16,
                        help='number of training data')   
    
    if study == 'Brain':
        parser.add_argument('--snr', type=float, default=10,
                            help='initial noise added for training')         
        parser.add_argument('--iterations', type=int, default=80,
                            help='number of iterations')
        parser.add_argument('--learning_rate', type=float, default=1e-3,
                            help='learning rate')
        parser.add_argument('--batchSize', type=int, default=4,
                            help='batch size')
        parser.add_argument('--patchSize', default=(64,64),
                            help='patch size')
        parser.add_argument('--tau', type=float, default=0.7,
                            help='tuning parameter for discrepancy principle')        
        parser.add_argument('--alpha', type=float, default=0.1,
                            help='tuning parameter for discrepancy principle') 
        if method == 'reside-s':
            parser.add_argument('--patches', type=int, default=576,
                                help='number of patches extracted from a image')
        if method == 'reside-m':
            parser.add_argument('--patches', type=int, default=144,
                                help='number of patches extracted from a image')
        
    if study == 'Perfusion':
        parser.add_argument('--snr', type=float, default=5,
                            help='initial noise added for training')
        parser.add_argument('--iterations', type=int, default=60,
                            help='number of iterations')
        parser.add_argument('--learning_rate', type=float, default=1e-4,
                            help='learning rate')     
        parser.add_argument('--batchSize', type=int, default=4,
                            help='batch size')
        parser.add_argument('--patchSize', default=(64,64,20),
                            help='patch size')
        parser.add_argument('--patches', type=int, default=288,
                            help='number of patches extracted from a image')
        parser.add_argument('--alpha', type=int, default=1,
                            help='tuning parameter for discrepancy principle') 
        if mode == 'MRXCAT':
            parser.add_argument('--tau', type=float, default=0.9,
                                help='tuning parameter for discrepancy principle')
        if mode == 'Perf':       
            parser.add_argument('--tau', type=float, default=1.1,
                                help='tuning parameter for discrepancy principle')

    return parser
