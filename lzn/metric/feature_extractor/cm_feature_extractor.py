"""Adapted from https://github.com/openai/consistency_models/blob/main/evaluations/evaluator.py"""

import numpy as np
import torch
import gc
import random
import os
import requests
import tensorflow.compat.v1 as tf
from tqdm import tqdm
from collections import namedtuple

from lzn.pytorch_utils.model_inference import to_uint8

FeatureExtractorResult = namedtuple("FeatureExtractorResult", ["pred", "spatial_pred", "softmax"])


class CMFeatureExtractor:
    def __init__(self, sess):
        super().__init__()
        self.sess = sess
        with self.sess.graph.as_default():
            self.image_input = tf.placeholder(tf.float32, shape=[None, None, None, 3])
            self.softmax_input = tf.placeholder(tf.float32, shape=[None, 2048])
            self.pool_features, self.spatial_features = _create_feature_graph(self.image_input)
            self.softmax = _create_softmax_graph(self.softmax_input)

    @torch.no_grad()
    def forward(self, x):
        device = x.device
        type_ = x.dtype
        x = x.cpu().detach().numpy()
        if x.shape[1] == 1:
            x = np.repeat(x, 3, axis=1)
        x = x.transpose((0, 2, 3, 1))
        x = to_uint8(x=x, min=-1, max=1)
        gc.collect()

        x = x.astype(np.float32)
        pred, spatial_pred = self.sess.run([self.pool_features, self.spatial_features], {self.image_input: x})
        pred = pred.reshape([pred.shape[0], -1])
        spatial_pred = spatial_pred.reshape([spatial_pred.shape[0], -1])

        softmax = self.sess.run(self.softmax, {self.softmax_input: pred})

        pred = torch.from_numpy(pred).to(device).to(type_)
        spatial_pred = torch.from_numpy(spatial_pred).to(device).to(type_)
        softmax = torch.from_numpy(softmax).to(device).to(type_)

        return FeatureExtractorResult(pred=pred, spatial_pred=spatial_pred, softmax=softmax)


INCEPTION_V3_URL = (
    "https://openaipublic.blob.core.windows.net/diffusion/jul-2021/ref_batches/classify_image_graph_def.pb"
)
INCEPTION_V3_PATH = f"/tmp/{random.randrange(2**32)}/classify_image_graph_def.pb"

FID_POOL_NAME = "pool_3:0"
FID_SPATIAL_NAME = "mixed_6/conv:0"


def _download_inception_model():
    if os.path.exists(INCEPTION_V3_PATH):
        return
    os.makedirs(os.path.dirname(INCEPTION_V3_PATH), exist_ok=True)
    print("downloading InceptionV3 model...")
    with requests.get(INCEPTION_V3_URL, stream=True) as r:
        r.raise_for_status()
        tmp_path = INCEPTION_V3_PATH + ".tmp"
        with open(tmp_path, "wb") as f:
            for chunk in tqdm(r.iter_content(chunk_size=8192)):
                f.write(chunk)
        os.rename(tmp_path, INCEPTION_V3_PATH)


def _create_feature_graph(input_batch):
    _download_inception_model()
    prefix = f"{random.randrange(2**32)}_{random.randrange(2**32)}"
    with open(INCEPTION_V3_PATH, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
    pool3, spatial = tf.import_graph_def(
        graph_def,
        input_map={"ExpandDims:0": input_batch},
        return_elements=[FID_POOL_NAME, FID_SPATIAL_NAME],
        name=prefix,
    )
    _update_shapes(pool3)
    spatial = spatial[..., :7]
    return pool3, spatial


def _update_shapes(pool3):
    # https://github.com/bioinf-jku/TTUR/blob/73ab375cdf952a12686d9aa7978567771084da42/fid.py#L50-L63
    ops = pool3.graph.get_operations()
    for op in ops:
        for o in op.outputs:
            shape = o.get_shape()
            if shape._dims is not None:  # pylint: disable=protected-access
                # shape = [s.value for s in shape] TF 1.x
                shape = [s for s in shape]  # TF 2.x
                new_shape = []
                for j, s in enumerate(shape):
                    if s == 1 and j == 0:
                        new_shape.append(None)
                    else:
                        new_shape.append(s)
                o.__dict__["_shape_val"] = tf.TensorShape(new_shape)
    return pool3


def _create_softmax_graph(input_batch):
    _download_inception_model()
    prefix = f"{random.randrange(2**32)}_{random.randrange(2**32)}"
    with open(INCEPTION_V3_PATH, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
    (matmul,) = tf.import_graph_def(graph_def, return_elements=["softmax/logits/MatMul"], name=prefix)
    w = matmul.inputs[1]
    logits = tf.matmul(input_batch, w)
    return tf.nn.softmax(logits)
