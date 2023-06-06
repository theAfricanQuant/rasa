import numpy as np
import typing
from typing import Any, List, Text

from rasa.nlu.config import RasaNLUModelConfig
from rasa.nlu.featurizers import Featurizer
from rasa.nlu.tokenizers import Token
from rasa.nlu.training_data import Message, TrainingData

if typing.TYPE_CHECKING:
    import mitie


class MitieFeaturizer(Featurizer):

    provides = ["text_features"]

    requires = ["tokens", "mitie_feature_extractor"]

    @classmethod
    def required_packages(cls) -> List[Text]:
        return ["mitie", "numpy"]

    def ndim(self, feature_extractor: "mitie.total_word_feature_extractor"):

        return feature_extractor.num_dimensions

    def train(
        self, training_data: TrainingData, config: RasaNLUModelConfig, **kwargs: Any
    ) -> None:

        mitie_feature_extractor = self._mitie_feature_extractor(**kwargs)
        for example in training_data.intent_examples:
            features = self.features_for_tokens(
                example.get("tokens"), mitie_feature_extractor
            )
            example.set(
                "text_features",
                self._combine_with_existing_text_features(example, features),
            )

    def process(self, message: Message, **kwargs: Any) -> None:

        mitie_feature_extractor = self._mitie_feature_extractor(**kwargs)
        features = self.features_for_tokens(
            message.get("tokens"), mitie_feature_extractor
        )
        message.set(
            "text_features",
            self._combine_with_existing_text_features(message, features),
        )

    def _mitie_feature_extractor(self, **kwargs):
        if mitie_feature_extractor := kwargs.get("mitie_feature_extractor"):
            return mitie_feature_extractor
        else:
            raise Exception(
                "Failed to train 'MitieFeaturizer'. "
                "Missing a proper MITIE feature extractor. "
                "Make sure this component is preceded by "
                "the 'MitieNLP' component in the pipeline "
                "configuration."
            )

    def features_for_tokens(
        self,
        tokens: List[Token],
        feature_extractor: "mitie.total_word_feature_extractor",
    ) -> np.ndarray:

        vec = np.zeros(self.ndim(feature_extractor))
        for token in tokens:
            vec += feature_extractor.get_feature_vector(token.text)
        return vec / len(tokens) if tokens else vec
