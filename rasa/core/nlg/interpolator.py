import re
import logging

logger = logging.getLogger(__name__)


def interpolate_text(template, values):
    if isinstance(template, str):
        # transforming template tags from
        # "{tag_name}" to "{0[tag_name]}"
        # as described here:
        # https://stackoverflow.com/questions/7934620/python-dots-in-the-name-of-variable-in-a-format-string#comment9695339_7934969
        # black list character and make sure to not to allow
        # (a) newline in slot name
        # (b) { or } in slot name
        try:
            text = re.sub(r"{([^\n{}]+?)}", r"{0[\1]}", template)
            text = text.format(values)
            return template.format({}) if "0[" in text else text
        except KeyError as e:
            logger.exception(
                f"Failed to fill utterance template '{template}'. Tried to replace '{e.args[0]}' but could not find a value for it. There is no slot with this name nor did you pass the value explicitly when calling the template. Return template without filling the template. "
            )
            return template
    return template


def interpolate(template, values):
    if isinstance(template, str):
        return interpolate_text(template, values)
    elif isinstance(template, dict):
        for k, v in template.items():
            if isinstance(v, dict):
                interpolate(v, values)
            else:
                template[k] = interpolate_text(v, values)
        return template
    return template
