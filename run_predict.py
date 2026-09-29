# -*- coding: utf-8 -*-
"""
@author: Gabriel Mittag, TU-Berlin
"""
import argparse

from nisqa.NISQA_model import nisqaModel


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', required=True, choices=('predict_file', 'predict_dir', 'predict_csv'))
    parser.add_argument('--pretrained_model', required=True, help='path to pretrained model')
    parser.add_argument('--deg', help='path to speech file')
    parser.add_argument('--data_dir', help='folder with speech files')
    parser.add_argument('--output_dir', help='folder to output NISQA_results.csv')
    parser.add_argument('--csv_file', help='CSV file name')
    parser.add_argument('--csv_deg', help='column in CSV with file names/paths')
    parser.add_argument('--csv_ref', default=argparse.SUPPRESS,
                        help='reference file column for double-ended CSV prediction')
    parser.add_argument('--num_workers', type=int, default=None,
                        help='DataLoader workers (default: up to 8; 0 for single files or legacy prediction)')
    parser.add_argument('--worker_threads', type=int, default=1,
                        help='BLAS/OpenMP threads per worker; BLAS threads with 0 workers (0: no cap)')
    parser.add_argument('--legacy_padding', action='store_true',
                        help='use original fixed padding and uncapped thread pools')
    parser.add_argument('--bs', type=int, default=1, help='batch size for predicting')
    parser.add_argument('--ms_channel', type=int, help='audio channel in case of stereo file')
    args = vars(parser.parse_args(argv))

    if args['mode'] == 'predict_file':
        if args['deg'] is None:
            raise ValueError('--deg argument with path to input file needed')
    elif args['mode'] == 'predict_dir':
        if args['data_dir'] is None:
            raise ValueError('--data_dir argument with folder with input files needed')
    elif args['mode'] == 'predict_csv':
        if args['csv_file'] is None:
            raise ValueError('--csv_file argument with csv file name needed')
        if args['csv_deg'] is None:
            raise ValueError('--csv_deg argument with csv column name of the filenames needed')
        if args['data_dir'] is None:
            args['data_dir'] = ''
    args['tr_bs_val'] = args['bs']
    args['tr_num_workers'] = args['num_workers']
    nisqa = nisqaModel(args)
    return nisqa.predict()


if __name__ == "__main__":
    main()
