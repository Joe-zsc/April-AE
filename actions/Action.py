from pathlib import Path
from tqdm import tqdm
import numpy as np
import pandas as pd
import logging
import time
import json
import sys
import os
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer
import os, sys
from annoy import AnnoyIndex

curr_path = os.path.dirname(__file__)
parent_path = os.path.dirname(curr_path)
sys.path.append(parent_path)  # add current terminal path to sys.path
sys.path.append(curr_path)  # add current terminal path to sys.path
from util import Configure, Action_Class
# from NLP_Module.sentence_embedding import sentence_embeddings
from NLP_Module.Encoder import encoder

class Action_embedding:
    action_description_path = Path("GatheredInfo/Action_description.json")
    with open(action_description_path, "r", encoding="utf-8") as f:  # *********
        action_description: dict = json.loads(f.read())
    use_action_embedding = True
    action_dim = encoder.SBERT_model_dim
    embedding_model = "SBERT"
    embedding_model_name = Configure.get("Embedding", f"{embedding_model}_model")

    # manhattan angular  euclidean
    nn_metric = "angular"
    use_dim_reduction = True
    normalization = True
    reduction_action_dim = 30

    def __init__(
        self,
        actions: list,
        action_path: Path,
        action_dim: int = action_dim,
        dim_reduction=use_dim_reduction,
        embedding_model=embedding_model,
        reduction_action_dim=reduction_action_dim,
        model_name=embedding_model_name,
        normalization=normalization,
    ):
        self.action_data = actions
        self.action_path = action_path
        # self.action_labels, self.action_labels_index = self.get_action_labels(
        #     all_label=False)
        self.action_dim = action_dim
        self.model_name = model_name
        self._low = np.array(-1)
        self._high = np.array(1)
        self._range = self._high - self._low

        self._space_low = -1
        self._space_high = 1
        self._k = (self._space_high - self._space_low) / self._range

        if dim_reduction and self.action_dim > reduction_action_dim:
            self.action_dim = reduction_action_dim
        self.normalization = normalization
        if normalization:
            self.action_vector_checkout = (
                self.action_path
                / f"Embedding-{self.action_dim}_dim-{embedding_model}-{model_name}-normalization.npy"
            )
        else:
            self.action_vector_checkout = (
                self.action_path
                / f"Embedding-{self.action_dim}_dim-{embedding_model}-{model_name}-Nonormalization.npy"
            )

        self.vector_space = None
        self.build_action_embedding()
        if self.action_dim != 1:
            if dim_reduction:
                self.LSA_dim_reduction(dim=self.action_dim)
            self.max_abs_action = self.get_max_abs_action()
            self.max_action = self.get_max_action()
            self.min_action = self.get_min_action()
        else:
            self.max_abs_action = 1.0
            self.max_action = 1.0
            self.min_action = -1.0

        self.max_abs_action = 1.0
        self.max_action = 1.0
        self.min_action = -1.0
        self.annoy_index = self.build_annoy()

    def build_annoy(self):

        num_vectors, vecrot_dim = self.vector_space.shape
        num_trees = int(np.log(num_vectors).round(0))
        index = AnnoyIndex(vecrot_dim, metric=self.nn_metric)
        print("building annoy index...")
        for i, vector in enumerate(tqdm(self.vector_space)):
            index.add_item(i, vector)
        index.build(
            num_trees, n_jobs=-1
        )  # n_trees 表示树的棵数，n_jobs 表示线程个数，n_jobs=-1 表示使用所有的 CPU 核；
        return index

    def search_annoy_neighbor(self, point, k_neighbor):
        # if not isinstance(point, np.ndarray):
        #     point = np.array([point]).astype(np.float32)
        if not isinstance(point, np.ndarray):
            point = np.array([point]).astype(np.float32)
        # search_res=[]
        # for v in point:
        search_res, distance = self.annoy_index.get_nns_by_vector(
            vector=point, n=k_neighbor, include_distances=True
        )
        search_res = np.array(search_res)
        if k_neighbor > 1:
            search_res = search_res.reshape(1, -1)
        # search_res.append(id)
        # search_res=np.array(search_res)
        knns = self.vector_space[search_res]
        p_out = []
        for p in knns:
            p_out.append(p)
        if k_neighbor == 1:
            p_out = [p_out]
            distance = distance[0]
        # p_out是实际环境可执行的动作
        return knns, np.array(p_out), search_res.tolist()[0], distance

        # ids=self.annoy_index.get_nns_by_vector(point.flatten(),k_neighbor)
        # vectors=self.vector_space[ids]
        # return vectors,vectors,ids

    def get_nearest_neighbor(self, point, k):

        return self.search_annoy_neighbor(point=point, k_neighbor=k)

    def build_action_embedding(self):
        if self.action_dim == 1:
            self.vector_space = self.init_uniform_space(
                low=[-1], high=[1], points=len(self.action_data)
            )
            return
        else:

            if not os.path.exists(self.action_vector_checkout):

                self.vector_space = self.init_SBERT_action_embedding()

                np.save(self.action_vector_checkout, self.vector_space)
            else:
                self.vector_space = np.load(self.action_vector_checkout)
            # return

    def get_max_abs_action(self):
        max_num = abs(np.max(self.vector_space))
        min_num = abs(np.min(self.vector_space))
        max_action = round(max(max_num, min_num) + 0.05, 1)
        return max_action

    def get_min_action(self):
        min_num = np.min(self.vector_space)
        min_action = round(min_num - 0.05, 1)
        return min_action

    def get_max_action(self):
        max_num = abs(np.max(self.vector_space))
        max_action = round(max_num + 0.05, 1)
        return max_action



    def init_SBERT_action_embedding(self):

        action_descriptions = []
        print("generating action embeddings...")
        for action in self.action_data:
            action_descriptions.append(self.get_action_description(action))
        # return sentence_embeddings(
        #     sentences=action_descriptions, normalization=self.normalization
        # )
        return  encoder.encode_SBERT(sentences=action_descriptions)

    def init_uniform_space(self, low, high, points):
        import itertools

        dims = 1
        # In Discrete situation, the action space is an one dimensional space, i.e., one row
        points_in_each_axis = round(points ** (1 / dims))

        axis = []
        for i in range(dims):
            axis.append(list(np.linspace(low[i], high[i], points_in_each_axis)))

        space = []
        for _ in itertools.product(*axis):
            space.append(list(_))
        a = np.array(space)
        # space: e.g., [[1], [2], ... ,[n-1]]
        return np.array(space).astype(np.float32)

    def get_SBERT_action_vector(self, action: str):
        des = self.get_action_description(action)
        # vector = sentence_embeddings(sentences=[des])
        vector = encoder.encode_SBERT(sentences=[des])
        return vector.flatten().astype(np.float32)


    def get_action_description(self, action):

        return self.action_description[action]

    def LSA_dim_reduction(self, dim=100):
        vectors = self.vector_space
        # svd=TruncatedSVD(n_components=100,random_state=2023)
        svd = make_pipeline(TruncatedSVD(n_components=dim), Normalizer(copy=False))
        embedding = svd.fit_transform(vectors)
        # embedding /= np.linalg.norm(embedding, axis=1, keepdims=True)
        self.vector_space = embedding
        return embedding


class Action:
    actions_file_path = Path(
        os.path.join(curr_path, Configure.get("common", "actions_file"))
    )
    with open(actions_file_path / "actions.json", "r") as f:  # *********
        actions_file = json.loads(f.read())
    exp_actions = actions_file["vuls"]

    exp_name_set = [v["name"] for v in exp_actions]
    All_EXP = [
        Action_Class(
            id=int(v["id"]),
            name=v["name"],
            act_cost=5,  # 10
            success_reward=1000,
            type="Exploit",
            config=v["exp_config"] if "exp_config" in v.keys() else "Unknown",
        )
        for v in exp_actions
    ]
    assert len(All_EXP) == len(
        set(All_EXP)
    ), f"{All_EXP-set(All_EXP)}"  # 检查vul set是否有重复元素
    PORT_SCAN = Action_Class(
        id=-1, name="Port Scan", act_cost=0, success_reward=0, type="Scan"
    )
    OS_SCAN = Action_Class(
        id=-2, name="OS Detect", success_reward=100, act_cost=3, type="Scan"
    )
    SERVICE_SCAN = Action_Class(
        id=-3, name="Service Scan", success_reward=100, act_cost=2, type="Scan"
    )
    PORT_SERVICE_SCAN = Action_Class(
        id=-4, name="Port&Service Scan", act_cost=2, type="Scan"
    )
    WEB_SCAN = Action_Class(
        id=-5, name="Web Scan", act_cost=5, success_reward=100, type="Scan"
    )

    # legal_actions = [PORT_SERVICE_SCAN, WEB_SCAN]
    Scan_actions = [PORT_SCAN, SERVICE_SCAN, OS_SCAN, WEB_SCAN]
    legal_actions = Scan_actions + All_EXP
    legal_actions_name = [action.name for action in legal_actions]
    action_space = len(legal_actions)
    if Action_embedding.use_action_embedding:
        action_embedding = Action_embedding(
            actions=legal_actions_name, action_path=actions_file_path
        )
    # types of errors
    # 1 重复动作执行
    action_repetition = dict(cost=10, message="action_repetition")

    # 2 前置动作未执行，信息依赖
    action_dependence = dict(cost=10, message="action_dependence")

    # 3 动作执行失败
    action_failed = dict(cost=10, message="action_failed")
    # Embedding = Action_embedding()

    def __init__(self):
        self.history_set = set()
        self.ExpActionFailedCount = (
            self.count_ExpAction()
        )  # Train_Real mode use it to speed up training process
        self.webscan_counts = 0
        self.exp_counts = 0
        self.last_action_id = -999

    def reset(self):
        self.history_set = set()
        self.webscan_counts = 0
        self.exp_counts = 0

    def count_ExpAction(self):
        vuls = []
        for v in self.All_EXP:
            vuls.append(v.name)
        count = dict.fromkeys(vuls, 0)
        return count

    def action_constraint(self, action):
        """
        return True if action is constrainted
        """
        if action.id in self.history_set:

            return self.action_repetition

        if action == self.SERVICE_SCAN:
            if self.PORT_SCAN.id not in self.history_set:
                return self.action_dependence
        if action == self.OS_SCAN or action in self.All_EXP:
            if not (
                self.PORT_SCAN.id in self.history_set
                or self.PORT_SERVICE_SCAN.id in self.history_set
            ):
                return self.action_dependence
        if action == self.WEB_SCAN:
            if not (
                self.SERVICE_SCAN.id in self.history_set
                or self.PORT_SERVICE_SCAN.id in self.history_set
            ):
                return self.action_dependence

        return None

    @classmethod
    def test_action(cls, action_mask):
        if type(action_mask) == int:
            if action_mask < 0 or action_mask > (cls.action_space):
                logging.error(
                    "legal actions include "
                    + ",".join([a.name for a in Action.legal_actions])
                )
                return False
            else:
                return True
        elif type(action_mask) == str:
            if action_mask in [a.name for a in Action.legal_actions]:
                return True
            else:
                logging.error(
                    "legal actions include "
                    + ",".join([a.name for a in Action.legal_actions])
                )
                return False
        else:
            return False

    @classmethod
    def get_action(cls, action_mask: int):
        action_name = cls.legal_actions[action_mask].name
        return action_name
