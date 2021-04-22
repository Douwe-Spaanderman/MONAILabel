import argparse
import distutils.util
import json
import logging
import os
import pathlib
import shutil
import sys

import yaml

from monailabel.utils.app_utils import app_instance

logger = logging.getLogger(__name__)


def test_inference(args):
    output_path = args.output
    os.makedirs(output_path, exist_ok=True)

    app = app_instance(app_dir=args.app_dir, studies=args.studies)
    logger.info('Running Inference: {}'.format(args.name))

    res_img = None
    res_json = None
    for it in range(args.runs):
        request = {
            "model": args.model,
            "image": args.input,
            "params": json.loads(args.params),
            "device": args.device
        }
        res_img, res_json = app.infer(request=request)

    if res_img:
        file_ext = ''.join(pathlib.Path(res_img).suffixes)
        result_image = os.path.join(args.output, 'label_' + args.model + file_ext)

        shutil.move(res_img, result_image)
        os.chmod(result_image, 0o777)
        print('Check Result file: {}'.format(result_image))

    print('Result JSON: {}'.format(res_json))


def test_train(args):
    output_path = os.path.dirname(args.output)
    os.makedirs(output_path, exist_ok=True)

    app = app_instance(app_dir=args.app_dir, studies=args.studies)
    logger.info('Running Training: {}'.format(args.name))

    request = {
        'output_dir': args.output,
        'data_list': args.input,
        'data_root': args.dataset_root,
        'device': args.device,
        'epochs': args.epochs,
        'amp': args.amp,
        'train': {},
        'val': {},
    }
    app.train(request)


def test_info(args):
    app = app_instance(app_dir=args.app_dir, studies=args.studies)
    info = app.info()

    class MyDumper(yaml.Dumper):
        def increase_indent(self, flow=False, indentless=False):
            return super(MyDumper, self).increase_indent(flow, False)

    yaml.dump(info, sys.stdout, Dumper=MyDumper, sort_keys=False, default_flow_style=False, width=120, indent=2)


def strtobool(val):
    return bool(distutils.util.strtobool(val))


def test_main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--debug', action='store_true')
    parser.add_argument('-a', '--app_dir', required=True)
    parser.add_argument('-s', '--studies', required=True)
    parser.add_argument('--device', default='cuda')

    subparsers = parser.add_subparsers(help='sub-command help')

    parser_a = subparsers.add_parser('infer', help='infer help')
    parser_a.add_argument('-m', '--model', required=True, help="Pre-Trained Model for inference")
    parser_a.add_argument('-i', '--input', required=True, help="Input Image file")
    parser_a.add_argument('-o', '--output', required=True, help="Output Label file")
    parser_a.add_argument('-p', '--params', default='{}', help="Input Params for inference")
    parser_a.add_argument('-r', '--runs', type=int, default=1, help="Number of times to run same inference")
    parser_a.set_defaults(test='infer')

    parser_b = subparsers.add_parser('train', help='train help')
    parser_b.add_argument('-i', '--input', required=True, help="Dataset JSON")
    parser_b.add_argument('-o', '--output', required=True, help="Train output folder")
    parser_b.add_argument('-r', '--dataset_root', required=True, help="Dataset ROOT Folder")
    parser_b.add_argument('-e', '--epochs', type=int, default=1, help="Number of epochs")
    parser_b.add_argument('--amp', type=strtobool, default='true', help="Use AMP")
    parser_b.set_defaults(test='train')

    parser_c = subparsers.add_parser('info', help='info help')
    parser_c.set_defaults(test='info')

    args = parser.parse_args()
    if not hasattr(args, 'test'):
        parser.print_usage()
        exit(-1)

    for arg in vars(args):
        print('USING:: {} = {}'.format(arg, getattr(args, arg)))
    print("")

    logging.basicConfig(
        level=(logging.DEBUG if args.debug else logging.INFO),
        format='[%(asctime)s.%(msecs)03d][%(levelname)5s](%(name)s) - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S')

    if args.test == 'infer':
        test_inference(args)
    elif args.test == 'train':
        test_train(args)
    elif args.test == 'info':
        test_info(args)
    else:
        parser.print_help()
