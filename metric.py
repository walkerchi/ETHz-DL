import torch 
import torch.nn.functional as F
from typing import Optional,List
from tqdm import tqdm
import dgl

def is_tensor_too_large(th:torch.Tensor, size:float=1):
    """
        Parameters
        ----------
            th:     torch.Tensor
            size:   float
                    the maximum size in GB
    """
    return th.storage().element_size() * th.storage().size() > size * 2**30

def knn_graph(emb:torch.FloatTensor, k:int=5, verbose:bool=False):
    """
        Parameters
        ----------
            emb:    torch.FloatTensor[n, e]
                    the embedding of each node
            k:      int
                    the number of neighbors 
            batch_size:int      default:64
                    in case of emb too large,
                    use a batched loader to compute iteratively
        Returns
        -------
            graph:  torch.LongTensor[n, k]
                    the edge for each node
    """  
    g = dgl.knn_graph(emb, k=k+1, algorithm="nn-descent").remove_self_loop()
    g.ndata['emb'] = emb 
    g.ndata[dgl.NID] = torch.arange(g.number_of_nodes()).long()
    g.apply_edges(lambda e:{"sim":F.cosine_similarity(e.src['emb'],e.dst['emb'])})
    g.ndata.pop('emb')
    def edge_fn(edges):
        return {'sim':edges.data['sim'], dgl.NID:edges.src[dgl.NID]}
    def node_fn(nodes):
        n_neighbors = nodes.mailbox[dgl.NID].shape[1]
        index = torch.argsort(nodes.mailbox['sim'],dim=1,descending=True)
        if n_neighbors > k:
            index = index[:, :k]
        elif n_neighbors == k:
            pass 
        else:
            raise Exception(f"Expect n_neighbors to be greater or equal than {k} but got {n_neighbors}")
        return {'knn':nodes.mailbox[dgl.NID].gather(1,index)}
    g.update_all(edge_fn, node_fn)
    indices = g.ndata['knn']
    return indices.long()

def jaccard_similarity(set1:torch.LongTensor, set2:torch.LongTensor, average:bool=True, verbose:bool=False):
    """
        Parameters
        ----------
            set1:   torch.LongTensor[n, k1]
            set2:   torch.LongTensor[n, k2]
            average: bool
                    if average, 
                        the return value will be float
                    else
                        the return value will be torch.FloatTensor[n]
        Returns
        -------
            jaccard_score: float or torch.FloatTensor[n] 
    """
    assert len(set1) == len(set2)
    l_scores = []
    if verbose:
        iterator = tqdm(range(len(set1)), desc="jaccard similarity", total=len(set1))
    else:
        iterator = range(len(set1))
    for i in iterator:
        _set1 = set(set1[i].tolist())
        _set2 = set(set2[i].tolist())
        jaccard = len(_set1.intersection(_set2)) / len(_set1.union(_set2))
        l_scores.append(jaccard)    
    jaccard_score = torch.tensor(l_scores)
    if average:
        jaccard_score = jaccard_score.mean()
    return jaccard_score

class KNNSimilarityMetric:
    """
        For each text, get the topk nearest text.
        And for the corresponding image, get the topk nearest images.
        Check their Jacard Similarity

        Parameters
        ----------
            text_emb:       torch.FloatTensor[n, dim]
            k:              int
                            topk neighbors
        Returns
        -------
    """
    def __init__(self, text_emb:torch.FloatTensor, k:int = 5, verbose:bool=False):
        self.text_knn = knn_graph(text_emb, k, verbose)
        self.k = k
        self.verbose = verbose
    def __call__(self, image_knn:torch.LongTensor):
        """
            Parameters
            ----------
                image_knn:      torch.LongTensor [n_text, topk_image]
                                topk_image should be greater than or equal to the `k`

            Returns
            -------
                float:          the KNN-similarity metric score
        """
        assert image_knn.shape[1] >= self.k, f"not enough k for image_knn, expected more than  {self.k}, but got {image_knn.shape[1]}"
        image_knn = image_knn[:, :self.k]
        assert self.text_knn.shape  == image_knn.shape
        return jaccard_similarity(self.text_knn, image_knn, verbose=self.verbose).item()


class TopKMetric:
    def __init__(self, text_mask:List[int], k:int = 5, verbose:bool=False):
        self.text_mask = torch.tensor(text_mask).long()
        self.k = k  
    def __call__(self, image_topk:torch.LongTensor):
        """
            Parameters
            ----------
                image_topk:         torch.LongTensor[n_text, topk_image]
            Returns
            -------
                float:              the topk metric score

        """
        assert image_topk.shape[1] >= self.k, f"not enough k for image_knn, expected more than  {self.k}, but got {image_topk.shape[1]}"
        assert image_topk.shape[0] == len(self.text_mask), f"expect the number of image equals to the text_mask"
        image_topk = image_topk[:, :self.k]
        
        topk_score = (image_topk == self.text_mask[:, None]).any(-1).float()

        return topk_score.mean().item()

if __name__ == '__main__':
    from dataset import CocoImage
    from dataset import CocoText
    from dataset import ImageLoader
    from model import CascadeCLIP,CLIP,FLIPImageEncoder,MobileNetImageEncoder
    import argparse
    import logging
    import os

    parser = argparse.ArgumentParser()
    parser.add_argument("--image-backend", type=str, default="mobilenet", choices=["mobilenet", "flip"])
    parser.add_argument("--metric", type=str, default="knn", choices=["knn","topk"])
    parser.add_argument("--image-batch", type=int, default=2)
    parser.add_argument("--text-batch", type=int, default=32)
    parser.add_argument("--k", nargs="+", type=int, default=[5,10,15,20,25])
    parser.add_argument("--n-layers", type=int, default=2, help="number of projection layer in mobilenet")
    parser.add_argument("--projection-path", type=str, default="./.mobilenet/project_coco_train_ep1_ly2.pt", help="weight file of projection layer in mobilenet")
    parser.add_argument("--p", type=float, default=0.5, help="mask rate in FLIP")
    parser.add_argument("--cuda", action="store_true")
    parser.add_argument("--use-clip",action="store_true")
    parser.add_argument("--log", type=str, default="metric.log")
    config = parser.parse_args()

    if not os.path.exists("./.log"):
        os.mkdir("./.log")
    logging.basicConfig(filename=f".log/{config.log}",
                     format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s-%(funcName)s',
                     level=logging.INFO)

    image_backend = {
        "flip":FLIPImageEncoder,
        "mobilenet":MobileNetImageEncoder
    }[config.image_backend]

    coco_image = CocoImage()
    coco_text  = CocoText()
    image_loader = ImageLoader(
        coco_image,
        batch_size=config.image_batch
    )
    if config.metric == "knn":
        text_loader = torch.utils.data.DataLoader(
            coco_text.first(),
            batch_size=config.text_batch,
            drop_last=False
        )

        def metric_cascade_clip(k=[10]):
            if config.use_clip:
                model = CLIP(
                    n_layers=config.n_layers,
                    image_backend=image_backend, 
                    p=config.p,
                    mobilenet_projection_path=config.projection_path
                )
            else:
                model = CascadeCLIP(
                    n_layers=config.n_layers,
                    image_backend=image_backend, 
                    p=config.p,
                    mobilenet_projection_path=config.projection_path
                )
            if config.cuda:
                model.cuda()
            if config.use_clip:
                indices, text_emb = model.topk_images(image_loader, text_loader, topk=25, return_index=True, return_text_emb=True, verbose=True)
            else:
                indices, text_emb = model.topk_images(image_loader, text_loader, topk=25, topm=200, return_index=True, return_text_emb=True, verbose=True)
            metrics = []
            for _k in k:
                metric_fn = KNNSimilarityMetric(text_emb,k=_k,verbose=False)
                metric    = metric_fn(indices)
                metrics.append(metric)
            return metrics
        def metric_clip(k=[10]):
            model = CLIP()
            if config.cuda:
                model.cuda()
            indices, text_emb = model.topk_images(image_loader, text_loader, topk=25, return_index=True, return_text_emb=True, verbose=True)
            metrics = []
            for _k in k:
                metric_fn = KNNSimilarityMetric(text_emb,k=_k,verbose=False)
                metric    = metric_fn(indices)
                metrics.append(metric)
            return metrics
    elif config.metric == "topk":
        text_loader = torch.utils.data.DataLoader(
            coco_text.flatten(),
            batch_size=config.text_batch,
            drop_last=False
        )
        def metric_cascade_clip(k=[10]):
            if config.use_clip:
                model = CLIP(
                    n_layers=config.n_layers,
                    image_backend=image_backend, 
                    p=config.p,
                    mobilenet_projection_path=config.projection_path
                )
            else:
                model = CascadeCLIP(
                    n_layers=config.n_layers,
                    image_backend=image_backend, 
                    p=config.p,
                    mobilenet_projection_path=config.projection_path
                )
            if config.cuda:
                model.cuda()
            if config.use_clip:
                indices = model.topk_images(image_loader, text_loader, topk=25, return_index=True,  verbose=True)
            else:
                indices = model.topk_images(image_loader, text_loader, topk=25, topm=125, return_index=True,  verbose=True)
            metrics = []
            for _k in k:
                metric_fn = TopKMetric(coco_text.mask,k=_k,verbose=False)
                metric    = metric_fn(indices)
                metrics.append(metric)
            return metrics
        def metric_clip(k=[10]):
            model = CLIP()
            if config.cuda:
                model.cuda()
            indices = model.topk_images(image_loader, text_loader, topk=25, return_index=True,  verbose=True)
            metrics = []
            for _k in k:
                metric_fn = TopKMetric(coco_text.mask,k=_k,verbose=False)
                metric    = metric_fn(indices)
                metrics.append(metric)
            return metrics
    logging.info("config")
    for k,v in config._get_kwargs():
        logging.info(f"{str(k):25}:{str(v):25}")
    logging.info("\n\n\n")
    knn_cascade_clip = metric_cascade_clip(config.k)
    logging.info(f"{config.metric}[CasCLIP]{knn_cascade_clip}")
    knn_clip = metric_clip(config.k)
    logging.info(f"{config.metric}[CLIP]{knn_clip}")
    
    logging.info("\n\n\n")
    for i, k in enumerate(config.k):
        logging.info(f"[{config.metric}({k})]CLIP:{knn_clip[i]} CascadeCLIP:{knn_cascade_clip[i]}")
