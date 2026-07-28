import argparse
import _init_paths
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = [8, 8]

from lib.test.analysis.plot_results import plot_results, print_results, print_per_sequence_results
from lib.test.evaluation import get_dataset, trackerlist

# trackers = []
# dataset_name = 'uav'
"""stark"""
# trackers.extend(trackerlist(name='stark_s', parameter_name='baseline', dataset_name=dataset_name,
#                             run_ids=None, display_name='STARK-S50'))
# trackers.extend(trackerlist(name='stark_st', parameter_name='baseline', dataset_name=dataset_name,
#                             run_ids=None, display_name='STARK-ST50'))
# trackers.extend(trackerlist(name='stark_st', parameter_name='baseline_R101', dataset_name=dataset_name,
#                             run_ids=None, display_name='STARK-ST101'))
"""TransT"""
# trackers.extend(trackerlist(name='TransT_N2', parameter_name=None, dataset_name=None,
#                             run_ids=None, display_name='TransT_N2', result_only=True))
# trackers.extend(trackerlist(name='TransT_N4', parameter_name=None, dataset_name=None,
#                             run_ids=None, display_name='TransT_N4', result_only=True))
"""pytracking"""
# trackers.extend(trackerlist('atom', 'default', None, range(0,5), 'ATOM'))
# trackers.extend(trackerlist('dimp', 'dimp18', None, range(0,5), 'DiMP18'))
# trackers.extend(trackerlist('dimp', 'dimp50', None, range(0,5), 'DiMP50'))
# trackers.extend(trackerlist('dimp', 'prdimp18', None, range(0,5), 'PrDiMP18'))
# trackers.extend(trackerlist('dimp', 'prdimp50', None, range(0,5), 'PrDiMP50'))
"""ostrack"""
# trackers.extend(trackerlist(name='litetrack', parameter_name='B9_cae_center_all_big_ep300', dataset_name=dataset_name,
#                             run_ids=300, display_name='litetrackB9'))
# trackers.extend(trackerlist(name='ostrack', parameter_name='vitb_384_mae_ce_32x4_ep300', dataset_name=dataset_name,
#                             run_ids=None, display_name='OSTrack384'))


# dataset = get_dataset(dataset_name)
# dataset = get_dataset('otb', 'nfs', 'uav', 'tc128ce')
# plot_results(trackers, dataset, 'LaSOT', merge_results=True, plot_types=('success', 'norm_prec'),
#              skip_missing_seq=False, force_evaluation=True, plot_bin_gap=0.05)
# print_results(trackers, dataset, dataset_name, merge_results=True, plot_types=('success', 'norm_prec', 'prec'))
# print_results(trackers, dataset, 'UNO', merge_results=True, plot_types=('success', 'prec'))
def main():
    parser = argparse.ArgumentParser(description='Analyze tracking results')
    parser.add_argument('--dataset', type=str, default='uav',
                        help='Dataset name (e.g., uav, lasot, otb)')
    parser.add_argument('--tracker_name', type=str, default='litetrack',
                        help='Tracker name')
    parser.add_argument('--parameter', type=str, default='B9_cae_center_all_big_ep300',
                        help='Parameter name of the tracker')
    parser.add_argument('--run_id', type=int, default=300,
                        help='Run ID (can be None)')
    parser.add_argument('--display_name', type=str, default='litetrackB9',
                        help='Display name in legends')
    parser.add_argument('--merge', action='store_true', default=True,
                        help='Merge results from multiple runs')
    parser.add_argument('--plot_types', nargs='+', default=['success', 'norm_prec', 'prec'],
                        help='Metrics to print (e.g., success prec norm_prec)')
    args = parser.parse_args()

    # 构建 tracker 列表（当前只添加一个，可按需扩展）
    trackers = trackerlist(
        name=args.tracker_name,
        parameter_name=args.parameter,
        dataset_name=args.dataset,
        run_ids=args.run_id,
        display_name=args.display_name
    )

    dataset = get_dataset(args.dataset)

    # 打印结果（也可以选择 plot_results）
    # report_name 使用数据集名
    # print_results( trackers, dataset, args.dataset,  merge_results=args.merge, plot_types=tuple(args.plot_types))

    plot_results(trackers, dataset, args.dataset, merge_results=True, plot_types=('success', 'norm_prec'), skip_missing_seq=False, force_evaluation=True, plot_bin_gap=0.05)

    # 指定其他数据集和参数
    # python analysis_results.py --dataset lasot --parameter B9_cae_center_all_ep300 --run_id 300 --display_name myTracker


if __name__ == '__main__':
    main()