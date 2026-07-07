import argparse
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class Point_Options():
    """This classification defines options used during both training and test time.
    It also implements several helper functions such as parsing, printing, and saving the options.
    It also gathers additional options defined in <modify_commandline_options> functions in both dataset classification and model classification.
    """

    def __init__(self):
        """Reset the classification; indicates the classification hasn't been initailized"""
        self.initialized = False

    def initialize(self, parser):
        """Define the common options that are used in both training and test."""
        # basic parameters
        parser.add_argument('--data_root', type=str, default=str(PROJECT_ROOT / 'dataset'), help='path to dataset root')
        parser.add_argument('--dataset', type=str, default='MSLCC', help='[MSLCC|GF3_3m|FUSAR]')
        parser.add_argument('--model_name', type=str, default='WaterPLNet', choices=['WaterPLNet'], help='model name')
        parser.add_argument('--experiment_name', type=str, default='Point', help='name of the experiment. It decides where to load datafiles, store samples and models')
        parser.add_argument('--save_path', type=str, default=str(PROJECT_ROOT / 'ckpt'), help='models and predictions are saved here')
        parser.add_argument('--data_inform_path', type=str, default=str(PROJECT_ROOT / 'datafiles'), help='path to train/val/test split files')
        parser.add_argument('--annotation_mode', type=str, default='original', choices=['original', 'random', 'noise', 'center_noise'], help='which point-label annotation set to use')
        parser.add_argument('--point_label_dir', type=str, default='point_label', help='point label folder name under each split')
        parser.add_argument('--mask_dir', type=str, default='mask', help='mask folder name under each split')
        parser.add_argument('--domain_dir', type=str, default='domain', help='domain folder name under each split')

        # model parameter
        parser.add_argument('--backbone', type=str, default='resnet18', help='which resnet')
        parser.add_argument('--out_stride', type=int, default=32, help='out_stride')
        parser.add_argument('--num_classes', type=int, default=2, help='classes')

        # train parameters
        parser.add_argument('--batch_size', type=int, default=64, help='input batch size')
        parser.add_argument('--pin', type=bool, default=True, help='pin_memory or not')

        parser.add_argument('--num_workers', type=int, default=4, help='number of workers')
        parser.add_argument('--img_size', type=int, default=256, help='image size')
        parser.add_argument('--in_channels', type=int, default=3, help='input channels')

        parser.add_argument('--num_epochs', type=int, default=150, help='num of epochs')
        parser.add_argument('--base_lr', type=float, default=0.001, help='base learning rate')
        parser.add_argument('--decay', type=float, default=5e-4, help='decay')
        parser.add_argument('--log_interval', type=int, default=60, help='how long to log, set yo 100 batch')
        parser.add_argument('--resume', type=bool, default=False, help='resume the saved checkpoint or not')
        parser.add_argument('--seed', type=int, default=42, help='random seed')
        parser.add_argument('--device', type=str, default='auto', help='device for single-process train/test: auto, cpu, cuda, cuda:0, ...')
        parser.add_argument('--checkpoint', type=str, default=None, help='checkpoint path or checkpoint filename under save_path/dataset/checkpoint')

        parser.add_argument('--seg_weight', type=float, default=1.0, help='weight for main segmentation loss')
        parser.add_argument('--penalty_weight', type=float, default=1.0, help='weight for penalty loss')
        parser.add_argument('--shadow_weight', type=float, default=0.0, help='weight for shadow loss')
        parser.add_argument('--psr_weight', type=float, default=0.0, help='weight for psr loss')
        parser.add_argument('--align_weight', type=float, default=0.0, help='weight for align loss')
        parser.add_argument('--energy_weight', type=float, default=0.0, help='weight for energy loss')
        parser.add_argument('--exp_weight', type=float, default=0.0, help='weight for energy loss')
        parser.add_argument('--con_weight', type=float, default=0.0, help='weight for energy loss')

        parser.add_argument(
        "--local_rank", "--local-rank",
        type=int,
        default=0,
        help="local rank for distributed training, passed by torch.distributed.launch"
            )
        self.initialized = True
        return parser

    def gather_options(self):
        """Initialize our parser with basic options(only once).
        Add additional model-specific and dataset-specific options.
        These options are defined in the <modify_commandline_options> function
        in model and dataset classes.
        """
        if not self.initialized:  # check if it has been initialized
            parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
            parser = self.initialize(parser)

        # get the basic options
        self.parser = parser
        return parser.parse_args()

    def parse(self):
        """Parse our options, create checkpoints directory suffix, and set up gpu device."""
        opt = self.gather_options()
        if opt.annotation_mode == 'random':
            opt.point_label_dir = 'point_label_random'
            opt.domain_dir = 'domain_random'
        elif opt.annotation_mode == 'noise':
            opt.point_label_dir = 'point_label_noise'
            opt.domain_dir = 'domain_noise'
        elif opt.annotation_mode == 'center_noise':
            opt.point_label_dir = 'point_label_center_noise'
            opt.domain_dir = 'domain_center_noise'
        self.opt = opt

        return self.opt


if __name__ == '__main__':
    opt = Point_Options().parse()
    print(opt)
