import logging
import json
import os
import typing
from typing import Any, List, Text, Optional

import rasa.utils.io

from rasa.core import utils
from rasa.core.actions.action import (
    ACTION_BACK_NAME,
    ACTION_LISTEN_NAME,
    ACTION_RESTART_NAME,
)
from rasa.core.constants import USER_INTENT_BACK, USER_INTENT_RESTART
from rasa.core.domain import Domain, InvalidDomain
from rasa.core.events import ActionExecuted
from rasa.core.policies.policy import Policy
from rasa.core.trackers import DialogueStateTracker

if typing.TYPE_CHECKING:
    from rasa.core.policies.ensemble import PolicyEnsemble


logger = logging.getLogger(__name__)


class MappingPolicy(Policy):
    """Policy which maps intents directly to actions.

    Intents can be assigned actions in the domain file which are to be
    executed whenever the intent is detected. This policy takes precedence over
    any other policy."""

    def __init__(self, priority: int = 3) -> None:
        """Create a new Mapping policy."""

        super(MappingPolicy, self).__init__(priority=priority)

    @classmethod
    def validate_against_domain(
        cls, ensemble: Optional["PolicyEnsemble"], domain: Optional[Domain]
    ) -> None:
        if not domain:
            return

        has_mapping_policy = ensemble is not None and any(
            isinstance(p, cls) for p in ensemble.policies
        )
        has_triggers_in_domain = any(
            "triggers" in properties
            for intent, properties in domain.intent_properties.items()
        )
        if has_triggers_in_domain and not has_mapping_policy:
            raise InvalidDomain(
                "You have defined triggers in your domain, but haven't "
                "added the MappingPolicy to your policy ensemble. "
                "Either remove the triggers from your domain or "
                "exclude the MappingPolicy from your policy configuration."
            )

    def train(
        self,
        training_trackers: List[DialogueStateTracker],
        domain: Domain,
        **kwargs: Any
    ) -> None:
        """Does nothing. This policy is deterministic."""

        pass

    def predict_action_probabilities(
        self, tracker: DialogueStateTracker, domain: Domain
    ) -> List[float]:
        """Predicts the assigned action.

        If the current intent is assigned to an action that action will be
        predicted with the highest probability of all policies. If it is not
        the policy will predict zero for every action."""

        prediction = [0.0] * domain.num_actions
        intent = tracker.latest_message.intent.get("name")
        if intent == USER_INTENT_RESTART:
            action = ACTION_RESTART_NAME
        elif intent == USER_INTENT_BACK:
            action = ACTION_BACK_NAME
        else:
            action = domain.intent_properties.get(intent, {}).get("triggers")

        if tracker.latest_action_name == ACTION_LISTEN_NAME:
            if action:
                idx = domain.index_for_action(action)
                if idx is None:
                    logger.warning(f"MappingPolicy tried to predict unknown action '{action}'.")
                else:
                    prediction[idx] = 1

            if any(prediction):
                logger.debug(
                    f"The predicted intent '{intent}' is mapped to  action '{action}' in the domain."
                )
        elif tracker.latest_action_name == action and action is not None:
            latest_action = tracker.get_last_event_for(ActionExecuted)
            assert latest_action.action_name == action
            if latest_action.policy == type(
                self
            ).__name__ or latest_action.policy.endswith(f"_{type(self).__name__}"):
                # this ensures that we only predict listen, if we predicted
                # the mapped action
                logger.debug(
                    f"The mapped action, '{action}', for this intent, '{intent}', was executed last so MappingPolicy is returning to action_listen."
                )

                idx = domain.index_for_action(ACTION_LISTEN_NAME)
                prediction[idx] = 1
            else:
                logger.debug(
                    f"The mapped action, '{action}', for this intent, '{intent}', was executed last, but it was predicted by another policy, '{latest_action.policy}', so MappingPolicy is notpredicting any action."
                )
        else:
            logger.debug(
                f"There is no mapped action for the predicted intent, '{intent}'."
            )
        return prediction

    def persist(self, path: Text) -> None:
        """Only persists the priority."""

        config_file = os.path.join(path, "mapping_policy.json")
        meta = {"priority": self.priority}
        rasa.utils.io.create_directory_for_file(config_file)
        utils.dump_obj_as_json_to_file(config_file, meta)

    @classmethod
    def load(cls, path: Text) -> "MappingPolicy":
        """Returns the class with the configured priority."""

        meta = {}
        if os.path.exists(path):
            meta_path = os.path.join(path, "mapping_policy.json")
            if os.path.isfile(meta_path):
                meta = json.loads(rasa.utils.io.read_file(meta_path))

        return cls(**meta)
