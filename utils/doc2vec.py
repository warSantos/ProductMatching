import os
import numpy as np
from gensim.models.doc2vec import Doc2Vec, TaggedDocument


class Doc2VecModel:

    # Carrega modelo Doc2Vec.
    def __init__(
        self, tokens, model_path=None, params=None, rebuild=False, save_model=True
    ):

        if model_path is not None and os.path.exists(model_path) and not rebuild:
            self.model = Doc2Vec.load(model_path)

        if params is not None:

            keys = list(params.keys())
            keys.remove("documents")
            keys.sort()
            sufix = "_".join([key + "-" + str(params[key]) for key in keys])
            model_path = "models/d2v_" + sufix + ".model"

            if os.path.exists(model_path):
                self.model = Doc2Vec.load(model_path)
            else:
                tagged_data = [ TaggedDocument(doc, tags=[str(i)]) for i, doc in enumerate(tokens) ]
                self.model = Doc2Vec(
                    documents = tagged_data,
                    vector_size = params["vector_size"],
                    dm = params["dm"],
                    window = params["window"],
                    epochs = params["epochs"],
                    dbow_words = params["dbow_words"],
                    workers = params["workers"]
                )
                self.model.save(model_path)
        else:
            tagged_data = [ TaggedDocument(doc, tags=[str(i)]) for i, doc in enumerate(tokens) ]
            self.model = Doc2Vec(documents=tagged_data, workers=15)

    # Transforma dados em vetor com o modelo Doc2Vec.
    def transform(self, n_docs):

        return np.array([self.model.dv[str(i)] for i in range(n_docs)])
