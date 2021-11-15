import argparse
import tensorflow as tf
import tf_pim_ops
import alexnet
import gnmt
import rnnt
import deepspeech2
import resnet50

parser = argparse.ArgumentParser()
parser.add_argument('-b','--batch_size', default=1, help="Input batch size", type=int)
parser.add_argument('-i','--iterations', default=10, help="Number of iterations for profiling", type=int)
parser.add_argument('-d','--dtype', default='fp16' , help="fp16 or fp32 execution")
parser.add_argument('-p','--profile', action="store_true", help="Enabled/Disable profiling")
parser.add_argument('-f','--functional_verify', action="store_true", help="Enabled/Disable Functional verification")

subparser = parser.add_subparsers(dest='model')
alexnet_parser = subparser.add_parser('alexnet')
gnmt_parser = subparser.add_parser('gnmt')
rnnt_parser = subparser.add_parser('rnnt')
deepspeech2_parser = subparser.add_parser('deepspeech2')
resnet50_parser = subparser.add_parser('resnet50')

gnmt_parser.add_argument('-l','--max_seq_length', default=100, help="Maximum sequence length of GNMT input", type=int)
deepspeech2_parser.add_argument('-l','--max_seq_length', default=50, help="Maximum sequence length of GNMT input", type=int)

deepspeech2_parser.add_argument('-m','--module', default='keras' , help="keras or pim_custom execution")
alexnet_parser.add_argument('-m','--module', default='keras' , help="keras or pim_custom execution")
resnet50_parser.add_argument('-m','--module', default='keras' , help="keras or pim execution")

args = parser.parse_args()

def DummyExecute():
    # Create some tensors
    a = tf.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    b = tf.constant([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    c = tf.matmul(a, b)

if __name__ == '__main__':
    print('User arguments {}'.format(args))
    tf_pim_ops.pim_init()

    DummyExecute()

    if args.model == 'alexnet' :
        print("Executing AlexNet model")
        alexnet.alexnet_model_run(args)
    elif args.model == 'gnmt' :
        print("Executing GNMT model")
        gnmt.gnmt_model_run(args)
    elif args.model == 'rnnt' :
        print("Executing RNNT model")
        rnnt.rnnt_model_run(args)
    elif args.model == 'deepspeech2' :
        print("Executing DeepSpeech2 model")
        deepspeech2.deepspeech2_model_run(args)
    elif args.model == 'resnet50' :
        print("Executing Resnet50 model")
        resnet50.resnet_model_run(args)

    tf_pim_ops.pim_deinit()
    