import numpy as np
import torch
import os
import tables
import h5py
import argparse
import imageio
from tqdm import tqdm


def preprocess_dataset(events, flow_paths, env_name, flow_idx_2_event_idx, flow_ts, event_ts, dt, n_split,
                       save_dir, chunk_idx):
    # Allocate cost volume tensor for pre-processing
    event_repr = torch.zeros((n_split, 2, 200, 200), dtype=torch.float)

    pols = events['p']
    xs = events['x']
    ys = events['y']
    ts = torch.from_numpy(event_ts.astype('float64'))

    mapping_file = open(os.path.join(save_dir, '{}_chunk{}_mapping.txt'.format(env_name, chunk_idx)), 'w')
    mapping_file.write('flow_idx\tstart_event_idx\tend_event_idx\tflow_start_ts\tflow_end_ts')
    for flow_idx, flow_path in enumerate(tqdm(flow_paths, desc='Pre-processing {}'.format(env_name))):
        data = imageio.imread(flow_path, format='PNG-FI')
        x_flow = torch.from_numpy((data[:, :, 0].astype(np.float64) - 2 ** 15) / 128.0)
        y_flow = torch.from_numpy((data[:, :, 1].astype(np.float64) - 2 ** 15) / 128.0)
        flow_mask = torch.from_numpy(data[:,:, 2].astype(np.bool))
        flow = torch.stack((x_flow, y_flow))

        # Find a range of events that are between former and latter grayscale images
        start_event_idx = flow_idx_2_event_idx[flow_idx]
        end_event_idx = flow_idx_2_event_idx[flow_idx+1]
        start_t = flow_ts[flow_idx]
        end_t = flow_ts[flow_idx+1]

        sel_pols = np.asarray(pols[start_event_idx:end_event_idx])
        sel_xs = np.asarray(xs[start_event_idx:end_event_idx])
        sel_ys = np.asarray(ys[start_event_idx:end_event_idx])
        sel_pols = torch.from_numpy(sel_pols.astype('int32'))
        sel_xs = torch.from_numpy(sel_xs.astype('float32'))
        sel_ys = torch.from_numpy(sel_ys.astype('float32'))

        sel_ts = ts[start_event_idx:end_event_idx]

        event_repr.zero_()
        sel_ts_norm = (n_split*(sel_ts-start_t)/(end_t-start_t)).int()
        assert (torch.min(sel_ts_norm) >= 0) & (torch.max(sel_ts_norm) < n_split), ''

        x0 = sel_xs.int()
        y0 = sel_ys.int()
        
        # This script currently represents polarities with only one channel similar to E-RAFT work 
        for xlim in [x0, x0+1]:
            for ylim in [y0, y0+1]:
                valid_mask = (xlim < 200) & (xlim >= 0) & (ylim < 200) & (ylim >= 0)
                interp_weights = 1 * (1 - (xlim.float()-sel_xs).abs()) \
                                 * (1 - (ylim.float()-sel_ys).abs())

                index = 200 * 200 * sel_ts_norm.long() + 200 * ylim.long() + xlim.long()

                # event_repr[:, 0].put_(index[mask], interp_weights[mask], accumulate=True)
                pol_mask = (sel_pols == 1)
                event_repr[:, 0].put_(index[pol_mask & valid_mask], interp_weights[pol_mask & valid_mask],
                                      accumulate=True)
                pol_mask = (sel_pols == 0)
                event_repr[:, 1].put_(index[pol_mask & valid_mask], interp_weights[pol_mask & valid_mask],
                                      accumulate=True)

        # Save event_repr to a directory
        torch.save({'event_repr': event_repr, 'flow': flow, 'flow_mask': flow_mask},
                   os.path.join(save_dir, '{}_chunk{}_{:06d}.pt'.format(env_name, chunk_idx, flow_idx)))
        # np.savez_compressed(os.path.join(save_dir, '{}_chunk{}_{:06d}'.format(env_name, chunk_idx, flow_idx)),
        #                     event_repr=event_repr.numpy(), flow=flow.numpy(), flow_mask=flow_mask.numpy())
        # Write corresponding indices of event represenation and grayscale images to a mapping file
        mapping_file.write('\n{}\t{}\t{}\t{}\t{}'.format(flow_idx, start_event_idx, end_event_idx,
                                                             flow_ts[flow_idx], flow_ts[flow_idx+1]))
    mapping_file.close()


# Generate a mapping between flow start time and event that happens right after the timestamp of flow start time
def get_flow_idx_2_event_idx(event_ts, flow_ts):
    assert event_ts[0] < flow_ts[0] or event_ts[-1] > flow_ts[-1],\
        'Flow timestamp is not within a range of event timestamps'

    mapping = np.zeros((flow_ts.size, ), dtype=np.int64)

    event_idx = np.searchsorted(event_ts, flow_ts[0])
    for flow_idx, each in enumerate(tqdm(flow_ts, desc='Construct index mapping')):
        while event_ts[event_idx] < each:
            event_idx += 1
        mapping[flow_idx] = event_idx
        if flow_idx != 0:
            assert mapping[flow_idx-1] != event_idx, \
                'No events between flow idx {} and {}!'.format(flow_idx-1, flow_idx)
    
    return mapping


if __name__ == '__main__':
    # Parser for setting parameters of the pre-processing script 
    parser = argparse.ArgumentParser(description='Preprocessing dataset script for fast data loading',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--save-dir', type=str, default='E:/EBAL_v10/train/savedir_test/V12',
                        help='Path to a directory that results will be saved')
    parser.add_argument('--dataset-dir', type=str,
                        default='E:/EBAL_v10/test/',
                        help='Path to dataset')
    parser.add_argument('--dt', type=int, default=1, help='Frame difference for computing a photometric loss')
    parser.add_argument('--n-split', type=int, default=10, help='Number of bins for events representation')
    args = parser.parse_args()

    env_name_file_path = os.path.join(args.dataset_dir, 'env_names.txt')
    env_names = []
    with open(env_name_file_path, 'r') as env_name_file:
        for line in env_name_file:
            env_names.append(line.strip())
    # np.int32 is sufficient to capture event idx, all sequence have 315k events
    event_data_file_paths = [os.path.join(args.dataset_dir, env_name, 'events.h5')
                             for env_name in env_names]
    flow_timestamp_paths = [os.path.join(args.dataset_dir, env_name, 'flow', 'forward_timestamps.txt')
                            for env_name in env_names]
    flow_dirs = [os.path.join(args.dataset_dir, env_name, 'flow', 'forward') for env_name in env_names]
    save_dir = os.path.join(args.save_dir, 'np_dt{}_tsplit{}'.format(args.dt, args.n_split))
    
    # Create a directory to save a dataset if it doesn't exist
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    print(save_dir)

    for env_name, event_data_file_path, \
        flow_timestamp_path, \
        flow_dir in zip(env_names, event_data_file_paths, flow_timestamp_paths, flow_dirs):
        event_data_file = h5py.File(event_data_file_path, 'r')
        events = event_data_file['events']
        event_rts = np.asarray(events['t'])
        offset = int(event_data_file['t_offset'][()])  # offset in microsecond
   
        num_flow = sum(1 for line in open(flow_timestamp_path, 'r'))    # number of flow + 1
        flow_ts = []
        with open(flow_timestamp_path, 'r') as flow_timestamp_file:
            for flow_idx, line in enumerate(flow_timestamp_file):
                if flow_idx == 0:
                    continue
                chunks = line.strip().split(',')
                start_ts, end_ts = int(chunks[0]), int(chunks[1])
                if not start_ts in flow_ts:
                    flow_ts.append(start_ts)
                if not end_ts in flow_ts:
                    flow_ts.append(end_ts)
        flow_ts = np.array(flow_ts, dtype=np.int64)
        flow_rts = flow_ts - offset

        # Positive timestamps with respect to the event offset imply that flows are recorded between events
        # np.diff(timestamps)>=0 implies that timestamps are in ascending order
        assert np.all(flow_rts > 0) and np.all(np.diff(flow_rts) >= 0), 'Invalid timestamps for flows'

        # Check for large jump between timestamps
        diff = np.diff(flow_rts)

        list_chopped_flow_rts = []
        start_idx = 0
        for each_end_idx in np.where(diff > 1.1e5)[0].tolist():
            list_chopped_flow_rts.append(flow_rts[start_idx:each_end_idx+1])
            start_idx = each_end_idx+1
        list_chopped_flow_rts.append(flow_rts[start_idx:])
        assert flow_rts.size == sum([each.size for each in list_chopped_flow_rts]), 'Invalid spliting'

        flow_paths = [path for path in os.listdir(flow_dir) if path.endswith('.png')]
        flow_paths = sorted(flow_paths, key=lambda x: int(x.split('.')[0]))
        flow_paths = [os.path.join(flow_dir, flow_path) for flow_path in flow_paths]
        
        # modif!
        flow_paths = flow_paths[:-1]

        list_chopped_flow_paths = []
        num_flow_paths_per_chunks = [each.size-1 for each in list_chopped_flow_rts]
        assert len(flow_paths) == sum(num_flow_paths_per_chunks), \
            'Invalid spliting {} and {}'.format(len(flow_paths), sum(num_flow_paths_per_chunks))
        from itertools import islice

        it = iter(flow_paths)
        list_chopped_flow_paths = [list(islice(it, 0, i)) for i in num_flow_paths_per_chunks]

        for chunk_idx, (chopped_flow_rts, chopped_flow_paths) in enumerate(
                zip(list_chopped_flow_rts, list_chopped_flow_paths)):
            mapping_path = os.path.join(args.dataset_dir, 'flow_2_event_idx_mapping',
                                        '{}_fw_flow_idx_2_event_idx_chunk{:01d}.npy'.format(env_name, chunk_idx))
            if not os.path.isfile(mapping_path):
                # modif!
                dir_exists = os.path.exists(os.path.join(args.dataset_dir, 'flow_2_event_idx_mapping'))
                if not dir_exists:
                    os.mkdir(os.path.join(args.dataset_dir, 'flow_2_event_idx_mapping'))
                print('Creating a mapping for chunk {}/{} of {}'.format(chunk_idx+1,
                                                                        len(list_chopped_flow_rts), env_name))
                flow_idx_2_event_idx = get_flow_idx_2_event_idx(event_rts, chopped_flow_rts)
                np.save(mapping_path[:-4], flow_idx_2_event_idx)
            else:
                print('Loading a mapping for chunk {}/{} of {}'.format(chunk_idx+1,
                                                                       len(list_chopped_flow_rts), env_name))
                flow_idx_2_event_idx = np.load(mapping_path)

            preprocess_dataset(events=events, flow_paths=chopped_flow_paths, env_name=env_name,
                               flow_idx_2_event_idx=flow_idx_2_event_idx, flow_ts=chopped_flow_rts, event_ts=event_rts,
                               dt=args.dt, n_split=args.n_split, save_dir=save_dir, chunk_idx=chunk_idx)

