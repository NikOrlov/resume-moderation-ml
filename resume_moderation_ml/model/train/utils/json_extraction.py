from itertools import chain
from typing import Any, Iterator
from resume_moderation_ml.model.train.utils import is_string, is_iterable, is_integer, is_mapping
from resume_moderation_ml.model.train.utils.;

def get_all_strings_from_json(data: Any) -> Iterator[str]:
    if is_mapping(data):
        for string in chain.from_iterable(get_all_strings_from_json(item) for item in data.values()):
            yield string
    elif is_iterable(data):
        for string in chain.from_iterable(get_all_strings_from_json(item) for item in data):
            yield string
    elif is_string(data):
        yield data


def search_strings_in_json(data: Any, query: Any) -> Iterator[str]:
    if data is None:
        return
    if query is None:
        for string in get_all_strings_from_json(data):
            yield string
        return

    if is_iterable(query):
        query = list(query)
        if len(query) == 0:
            raise ValueError('Error searching in json. There is no sense in empty iterable query')
        for string in chain.from_iterable(search_strings_in_json(data, item) for item in query):
            yield string
        return

    if is_string(query) or is_integer(query):
        query = {query: None}

    if not is_mapping(query):
        raise ValueError('Error searching in json. Query must be string, iterable or mapping, got {}'.format(query))

    # now query contains mapping, so data must contain mapping, because we can extract keys only from mappings
    if not (is_mapping(data) or not is_integer(data)):
        return

    for key, sub_query in query.items():
        if is_integer(key):
            if not is_iterable(data):
                raise ValueError('Error searching in json. Number index could by applied only to iterable')
            data = list(data)
            if key >= len(data):  # nothing to select
                continue
            selected_data = data[key]
        else:
            if is_iterable(data):
                for string in chain.from_iterable(search_strings_in_json(item, {key: sub_query}) for item in data):
                    yield string
                continue
            selected_data = data.get(key)

        for string in search_strings_in_json(selected_data, sub_query):
            yield string

class JsonTextExtractor(BaseEstimator, TransformerMixin, NoFitMixin):
    def __init__(self, query):
        self.query = query

    def transform(self, X):
        return [u' '.join(search_strings_in_json(data_object, self.query)) for data_object in X]


# this functionality is implemented as callable class only for pickling
class FieldLengthExtractor(object):
    def __init__(self, field_name, required=True):
        self.field_name = field_name
        self.required = required

    def __call__(self, doc):
        if self.required and self.field_name not in doc:
            raise KeyError(self.field_name)
        value = doc.get(self.field_name)
        return len(value) if value else 0


def make_field_length_extractor(field_name, required=True):
    return ValueExtractor(FieldLengthExtractor(field_name, required), dtype=np.float64)

