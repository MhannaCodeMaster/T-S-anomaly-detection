import torch

from triplet.triplet import *
from triplet.tl_eval import *
from data.datasets import *
from data.data_utils import *


def main():
    args = load_args()
    triplet = TripletEmbedder(pretrained=False)
    triplet.cuda()
    val_loader, _, _ = load_val_crops(args)
    triplet.load_state_dict(torch.load(args.model_path)['state_dict'])
    emd, label = extract_embeddings(triplet, val_loader)
    
    save_path = "gallery_embeddings.pt"
    torch.save({
        "embeddings": emd,    # float32, [N, D]
        "labels": label       # long, [N]
    }, save_path)



if __name__ == '__main__':
    main()