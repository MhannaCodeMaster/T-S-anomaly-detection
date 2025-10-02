import torch

from src.triplet.triplet import *
from src.triplet.tl_eval import *
from src.data.datasets import *
from src.data.data_utils import *


def main():
    args = load_args()
    triplet = TripletEmbedder(pretrained=False)
    triplet.cuda()
    _, _, _, ok_loader = load_crops(args)
    triplet.load_state_dict(torch.load(args.model_path)['state_dict'])
    emd, label = extract_embeddings(triplet, ok_loader)
    
    save_path = "gallery_embeddings.pt"
    torch.save({
        "embeddings": emd,    # float32, [N, D]
        "labels": label       # long, [N]
    }, save_path)



if __name__ == '__main__':
    main()