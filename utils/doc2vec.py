import os
import numpy as np
from gensim.models.doc2vec import Doc2Vec


class Doc2VecModel:

    # Carrega modelo Doc2Vec.
    def __init__(
        self, tokens, model_path=None, params=None, rebuild=False, save_model=True
    ):

        if model_path is not None and os.path.exists(model_path) and not rebuild:
            self.model = Doc2Vec.load(model_path)

        if params is not None:

            keys = list(params.keys())
            keys.sort()
            sufix = "_".join([key + "-" + str(params[key]) for key in keys])
            model_path = "models/w2v_" + sufix + ".model"

            if os.path.exists(model_path):
                self.model = Doc2Vec.load(model_path)
            else:
                self.model = Doc2Vec(
                    sentences=tokens,
                    vector_size=params["vector_size"],
                    epochs=params["epochs"],
                )
                self.model.save(model_path)
        else:
            self.model = Doc2Vec(sentences=tokens, workers=15)

    # Transforma dados em vetor com o modelo Doc2Vec.
    def d2v_transform(self, n_docs, d2v):

        return np.array([d2v.docvecs[v] for v in range(n_docs)])
