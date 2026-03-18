from os.path import isfile
from typing import Union, Dict, List, Tuple, Any, Optional

from ovos_bus_client.message import Message
from ovos_bus_client.session import SessionManager
from ovos_plugin_manager.templates.pipeline import IntentHandlerMatch
from ovos_plugin_manager.templates.transformers import IntentTransformer
from ovos_utils.bracket_expansion import expand_template
from ovos_utils.lang import standardize_lang_tag
from ovos_utils.list_utils import deduplicate_list, flatten_list
from ovos_utils.log import LOG

from ahocorasick_ner import AhocorasickNER


class AhocorasickNERTransformer(IntentTransformer):
    """
    An OpenVoiceOS Intent Transformer plugin that performs Named Entity Recognition
    using the Aho-Corasick algorithm. It listens for entity registrations and
    injects matched entities into the intent match data.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initializes the transformer.

        Args:
            config: Optional configuration dictionary.
        """
        super().__init__("ovos-ahocorasick-ner-plugin", 5, config)
        self.matchers: Dict[str, Dict[str, AhocorasickNER]] = {}

    def bind(self, bus: Any) -> None:
        """
        Binds the transformer to the message bus and sets up event listeners.

        Args:
            bus: The message bus instance.
        """
        super().bind(bus)
        self.bus.on('padatious:register_entity', self.handle_register_entity)

    def _unpack_object(self, message: Message) -> Tuple[str, str, str, List[str], List[str]]:
        """
        Extracts and prepares training data from a registration message.

        Args:
            message: The registration message.

        Returns:
            Tuple: (lang, skill_id, entity_name, samples, blacklisted_words)
        """
        # standard info
        sess = SessionManager.get(message)
        skill_id = message.data.get("skill_id") or message.context.get("skill_id")
        if not skill_id:
            skill_id = "anonymous_skill"
        lang = message.data.get('lang') or sess.lang
        lang = standardize_lang_tag(lang)

        # intent specific
        file_name = message.data.get('file_name')
        samples = message.data.get("samples")
        name = message.data['name']
        blacklisted_words = message.data.get('blacklisted_words', [])
        
        if (not file_name or not isfile(file_name)) and not samples:
            raise FileNotFoundError(f'Could not find file or samples for entity {name}')
            
        if not samples and file_name and isfile(file_name):
            with open(file_name, 'r') as f:
                samples = [line.strip() for line in f.readlines()]

        # expand templates
        if samples:
            samples = deduplicate_list(flatten_list([expand_template(s) for s in samples]))
        else:
            samples = []

        return lang, skill_id, name, samples, blacklisted_words

    def handle_register_entity(self, message: Message) -> None:
        """
        Handles 'padatious:register_entity' messages to populate the NER matchers.

        Args:
            message: The registration message.
        """
        try:
            lang, skill_id, entity_name, samples, _ = self._unpack_object(message)
            if not samples or not entity_name:
                LOG.warning(f"Skipping entity registration: empty samples or entity_name for {skill_id}")
                return
            if lang not in self.matchers:
                self.matchers[lang] = {}
            if skill_id not in self.matchers[lang]:
                self.matchers[lang][skill_id] = AhocorasickNER()
            for s in samples:
                self.matchers[lang][skill_id].add_word(entity_name, s)
            LOG.debug(f"Registered {len(samples)} keywords for '{skill_id}:{entity_name}' ({lang})")
        except Exception as e:
            LOG.error(f"Failed to register entity: {e}")

    def transform(self, intent: IntentHandlerMatch) -> IntentHandlerMatch:
        """
        Transforms the intent match data by performing NER and injecting matches.

        Args:
            intent: The current intent handler match.

        Returns:
            IntentHandlerMatch: The transformed (or original) intent match.
        """
        try:
            sess = intent.updated_session or SessionManager.get()
            matchers = self.matchers.get(sess.lang)
            if matchers:
                skill_id = intent.skill_id or (
                    intent.match_type.split(":")[0]
                    if ":" in intent.match_type
                    else intent.match_type
                )
                if skill_id in matchers:
                    entities = {
                        e["label"]: e["word"]
                        for e in matchers[skill_id].tag(intent.utterance)
                    }
                    if entities:
                        LOG.debug(f"{skill_id} keywords match: {entities}")
                        intent.match_data.update(entities)
        except Exception as e:
            LOG.error(f"Error in AhocorasickNERTransformer.transform: {e}")
        return intent
