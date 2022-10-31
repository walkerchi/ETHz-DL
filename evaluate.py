import time
import logging
import numpy as np
from utils import load_model, load_dataset

def cos_sim(a, b):
    return np.dot(a, b)/(np.linalg.norm(a)*np.linalg.norm(b))

def get_top_k_accs(img_vecs, cap_vecs, which_top_k_accs):
    top_k_correct = {k: 0 for k in which_top_k_accs}
    for i, cap_vec in enumerate(cap_vecs):
        scores = [(j, cos_sim(img_vec, cap_vec)) for j, img_vec in enumerate(img_vecs)]
        scores.sort(key=lambda x: x[1], reverse=True)
        ranking = [score[0] for score in scores]
        for k in which_top_k_accs:
            if i in ranking[:k]:
                top_k_correct[k] += 1
    for k in which_top_k_accs:
        top_k_correct[k] /= len(cap_vecs)
    return top_k_correct


def top_k_accs_refined(img_vecss, cap_vecss, which_top_k_accs, top_k_small_preds_refine):
    top_k_correct = {k: 0 for k in which_top_k_accs}
    assert top_k_small_preds_refine >= max(which_top_k_accs)
    for i, cap_vec in enumerate(cap_vecss[0]):
        # Phase one: Score with small model
        scores = [(j, cos_sim(img_vec, cap_vec)) for j, img_vec in enumerate(img_vecss[0])]
        scores.sort(key=lambda x: x[1], reverse=True)
        # Phase two: Score with big model
        top_score_idxs = [x[0] for x in scores[:top_k_small_preds_refine]]
        scores = []
        for idx in top_score_idxs:
            scores.append((idx, cos_sim(img_vecss[1][idx], cap_vecss[1][i])))
        scores.sort(key=lambda x: x[1], reverse=True)
        ranking = [score[0] for score in scores]
        for k in which_top_k_accs:
            if i in ranking[:k]:
                top_k_correct[k] += 1
    for k in which_top_k_accs:
        top_k_correct[k] /= len(cap_vecss[0])
    return top_k_correct

def evaluate(config):
    logging.info("loading small model...")
    small_model = load_model(config["small_model"], config["small_model_args"])
    logging.info("loading big model...")
    big_model = load_model(config["big_model"], config["big_model_args"])
    if config["eval_only_small_model"]:
        models = [small_model]
    else:
        models = [small_model, big_model]
    logging.info("loading dataset...")
    dataset = load_dataset(config["dataset"], config["dataset_args"])
    imgs = dataset.images
    caps = dataset.captions
    img_times = []
    cap_times = []
    img_vecss = []
    cap_vecss = []
    for model in models:
        # Compute image embeddings
        logging.info("computing image embeddings...")
        img_time_start = time.time()
        img_vecss.append(model.img_vecs(imgs))
        img_times.append(time.time() - img_time_start)
        # Compute caption embeddings
        logging.info("computing caption embeddings...")
        cap_time_start = time.time()
        cap_vecss.append(model.cap_vecs(caps))
        cap_times.append(time.time() - cap_time_start)
    # Compute image retrieval accuracies
    logging.info("computing image retrieval accuracies...")
    top_k_accss = []
    for img_vecs, cap_vecs in zip(img_vecss, cap_vecss):
        top_k_accss.append(
            get_top_k_accs(img_vecs, cap_vecs, config["which_top_k_accs"]))
    # Compute refined image retrieval accuracies
    if not config["eval_only_small_model"]:
        top_k_accss.append(top_k_accs_refined(
                img_vecss,
                cap_vecss,
                config["which_top_k_accs"],
                config["top_k_small_preds_refine"]))
    # Report results
    results = {
        "processing times": {"images": img_times, "captions": cap_times},
        "top_k_accs": top_k_accss
    }
    logging.info(results)
