from galaxy import model
from galaxy.util.odict import odict
from .type_description import COLLECTION_TYPE_DESCRIPTION_FACTORY


def build_collection(type, dataset_instances, fields=None):
    """
    Build DatasetCollection with populated DatasetcollectionElement objects
    corresponding to the supplied dataset instances or throw exception if
    this is not a valid collection of the specified type.
    """
    dataset_collection = model.DatasetCollection(fields=fields)
    set_collection_elements(dataset_collection, type, dataset_instances, fields=fields)
    return dataset_collection


def set_collection_elements(dataset_collection, type, dataset_instances, fields=None):
    element_index = 0
    elements = []
    if fields == "auto":
        fields = guess_fields(dataset_instances)
    for element in type.generate_elements(dataset_instances, fields=fields):
        element.element_index = element_index
        element.collection = dataset_collection
        elements.append(element)

        element_index += 1

    dataset_collection.elements = elements
    dataset_collection.element_count = element_index
    return dataset_collection


def guess_fields(dataset_instances):
    fields = []
    for identifier, element in dataset_instances.items():
        # TODO: Make generic enough to handle nested record types.
        assert element.history_content_type == "dataset"
        fields.append({"class": "File", "name": identifier})

    return fields


class CollectionBuilder(object):
    """ Purely functional builder pattern for building a dataset collection. """

    def __init__(self, collection_type_description):
        self._collection_type_description = collection_type_description
        self._current_elements = odict()

    def replace_elements_in_collection(self, template_collection, replacement_dict):
        self._current_elements = self._replace_elements_in_collection(
            template_collection=template_collection,
            replacement_dict=replacement_dict,
        )

    def _replace_elements_in_collection(self, template_collection, replacement_dict):
        elements = odict()
        for element in template_collection.elements:
            if element.is_collection:
                collection_builder = CollectionBuilder(
                    collection_type_description=self._collection_type_description.child_collection_type_description()
                )
                collection_builder.replace_elements_in_collection(
                    template_collection=element.child_collection,
                    replacement_dict=replacement_dict
                )
                elements[element.element_identifier] = collection_builder
            else:
                elements[element.element_identifier] = replacement_dict.get(element.element_object, element.element_object)
        return elements

    def get_level(self, identifier):
        if not self._nested_collection:
            message_template = "Cannot add nested collection to collection of type [%s]"
            message = message_template % (self._collection_type_description)
            raise AssertionError(message)
        if identifier not in self._current_elements:
            subcollection_builder = CollectionBuilder(
                self._subcollection_type_description
            )
            self._current_elements[identifier] = subcollection_builder

        return self._current_elements[identifier]

    def add_dataset(self, identifier, dataset_instance):
        self._current_elements[identifier] = dataset_instance

    def build_elements(self):
        elements = self._current_elements
        if self._nested_collection:
            new_elements = odict()
            for identifier, element in elements.items():
                new_elements[identifier] = element.build()
            elements = new_elements
        return elements

    def build(self):
        type_plugin = self._collection_type_description.rank_type_plugin()
        collection = build_collection(type_plugin, self.build_elements())
        collection.collection_type = self._collection_type_description.collection_type
        return collection

    @property
    def _subcollection_type_description(self):
        return self._collection_type_description.subcollection_type_description()

    @property
    def _nested_collection(self):
        return self._collection_type_description.has_subcollections()


class BoundCollectionBuilder(CollectionBuilder):
    """ More stateful builder that is bound to a particular model object. """

    def __init__(self, dataset_collection):
        self.dataset_collection = dataset_collection
        if dataset_collection.populated:
            raise Exception("Cannot reset elements of an already populated dataset collection.")
        collection_type = dataset_collection.collection_type
        collection_type_description = COLLECTION_TYPE_DESCRIPTION_FACTORY.for_collection_type(collection_type)
        super(BoundCollectionBuilder, self).__init__(collection_type_description)

    def populate(self):
        elements = self.build_elements()
        type_plugin = self._collection_type_description.rank_type_plugin()
        set_collection_elements(self.dataset_collection, type_plugin, elements)
        self.dataset_collection.mark_as_populated()
