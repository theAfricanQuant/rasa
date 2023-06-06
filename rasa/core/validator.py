import logging
import asyncio
from typing import List
from rasa.core.domain import Domain
from rasa.importers.importer import TrainingDataImporter
from rasa.nlu.training_data import TrainingData
from rasa.core.training.dsl import StoryStep
from rasa.core.training.dsl import UserUttered
from rasa.core.training.dsl import ActionExecuted
from rasa.core.actions.action import UTTER_PREFIX

logger = logging.getLogger(__name__)


class Validator(object):
    """A class used to verify usage of intents and utterances."""

    def __init__(self, domain: Domain, intents: TrainingData, stories: List[StoryStep]):
        """Initializes the Validator object. """

        self.domain = domain
        self.intents = intents
        self.stories = stories

    @classmethod
    async def from_importer(cls, importer: TrainingDataImporter) -> "Validator":
        """Create an instance from the domain, nlu and story files."""

        domain = await importer.get_domain()
        stories = await importer.get_stories()
        intents = await importer.get_nlu_data()

        return cls(domain, intents, stories.story_steps)

    def verify_intents(self):
        """Compares list of intents in domain with intents in NLU training data."""

        domain_intents = set(self.domain.intent_properties)
        nlu_data_intents = {
            intent.data["intent"] for intent in self.intents.intent_examples
        }
        for intent in domain_intents:
            if intent not in nlu_data_intents:
                logger.warning(
                    f"The intent '{intent}' is listed in the domain file, but is not found in the NLU training data."
                )

        for intent in nlu_data_intents:
            if intent not in domain_intents:
                logger.error(
                    f"The intent '{intent}' is in the NLU training data, but is not listed in the domain."
                )

        return domain_intents

    def verify_intents_in_stories(self):
        """Checks intents used in stories.

        Verifies if the intents used in the stories are valid, and whether
        all valid intents are used in the stories."""

        domain_intents = self.verify_intents()

        stories_intents = set()
        for story in self.stories:
            for event in story.events:
                if type(event) == UserUttered:
                    intent = event.intent["name"]
                    stories_intents.add(intent)
                    if intent not in domain_intents:
                        logger.error(
                            f"The intent '{intent}' is used in stories, but is not listed in the domain file."
                        )

        for intent in domain_intents:
            if intent not in stories_intents:
                logger.warning(f"The intent '{intent}' is not used in any story.")

    def verify_utterances(self):
        """Compares list of utterances in actions with utterances in templates."""

        actions = self.domain.action_names
        valid_utterances = set()

        utterance_templates = set(self.domain.templates)
        for utterance in utterance_templates:
            if utterance in actions:
                valid_utterances.add(utterance)
            else:
                logger.error(
                    f"The utterance '{utterance}' is not listed under 'actions' in the domain file."
                )

        for action in actions:
            if action.startswith(UTTER_PREFIX):
                if action not in utterance_templates:
                    logger.error(f"There is no template for utterance '{action}'.")

        return valid_utterances

    def verify_utterances_in_stories(self):
        """Verifies usage of utterances in stories.

        Checks whether utterances used in the stories are valid,
        and whether all valid utterances are used in stories."""

        valid_utterances = self.verify_utterances()
        stories_utterances = set()

        for story in self.stories:
            for event in story.events:
                if isinstance(event, ActionExecuted) and event.action_name.startswith(
                    UTTER_PREFIX
                ):
                    utterance = event.action_name
                    if (
                        utterance not in valid_utterances
                        and utterance not in stories_utterances
                    ):
                        logger.error(
                            f"The utterance '{utterance}' is used in stories, but is not a valid utterance."
                        )
                    stories_utterances.add(utterance)

        for utterance in valid_utterances:
            if utterance not in stories_utterances:
                logger.warning(f"The utterance '{utterance}' is not used in any story.")

    def verify_all(self):
        """Runs all the validations on intents and utterances."""

        logger.info("Validating intents...")
        self.verify_intents_in_stories()

        logger.info("Validating utterances...")
        self.verify_utterances_in_stories()
