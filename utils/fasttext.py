import os
import numpy as np
from gensim.models.fasttext import FastText


class FastTextModel:

    # Carrega modelo FastText.
    def __init__(
        self, tokens, model_path=None, params=None, rebuild=False, save_model=True
    ):

        if model_path is not None and os.path.exists(model_path) and not rebuild:
            self.model = FastText.load(model_path)

        if params is not None:

            keys = list(params.keys())
            keys.remove("sentences")
            keys.sort()
            sufix = "_".join([key + "-" + str(params[key]) for key in keys])
            model_path = "models/fasttext_" + sufix + ".model"

            if os.path.exists(model_path):
                self.model = FastText.load(model_path)
            else:
                self.model = FastText(
                    sentences = params["sentences"],
                    vector_size = params["vector_size"],
                    sg = params["sg"],
                    window = params["window"],
                    epochs = params["epochs"],
                    workers = params["workers"]
                )
                self.model.save(model_path)
        else:
            self.model = FastText(sentences=tokens, workers=15)

    # Transforma dados em vetor com o modelo fasttext.
    def transform(self, sentences):

        vecs = []
        for s in sentences:
            vecs_t = []
            for token in s:
                if token in self.model.wv:
                    vecs_t.append(self.model.wv[token])
            if vecs_t:
                vecs.append(np.mean(vecs_t, axis=0))
            else:
                vecs.append(np.zeros(self.model.vector_size))
        return np.array(vecs)
