from torch.utils.data import DataLoader
from lib.train.dataset.vast_track import VastTrack

dataset = VastTrack(root='/path/to/root')
loader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=0)
for i, batch in enumerate(loader):
    print(f"Batch {i}: images shape {batch[0].shape}, bbox shape {batch[1]['bbox'].shape}")
    if i >= 2:
        break