import faiss
import numpy as np
import torch
from transformers import BertTokenizer, BertModel

class vector_Database:
    def __init__(self):
        self.index = None # the index with ID
        self.base_index = None # the index without ID
        # TODO may need to adjust the structure of this part. 6/6/2025
        self.text_Path = None
        self.index_Path = None
        self.dimension = None

    def set_base_index(self, index):
        """
        Function for setting the base_index
        Do not use this function directly
        (optional)
        :param index: base_index
        :return: None
        """
        self.base_index = index

    def set_index(self, index):
        """
        Function for setting the index
        :param index:  index
        :return: None
        """
        self.index = index

    def __transform(self, text, model_name):
        """
        Function for transform words to vector
        One sentence each time
        :param text: input sentence
        :param model_name: the model used to transform the sentence
        :return:
        """
        tokenizer = BertTokenizer.from_pretrained(model_name)
        model = BertModel.from_pretrained(model_name)

        inputs = tokenizer(text, return_tensors='pt')

        with torch.no_grad():
            outputs = model(**inputs)
            cls_vector = outputs.last_hidden_state[:, 0, :]
            return cls_vector

    def transform_word_to_vector(self, text:list, model):
        """
        The function for transforming the list of sentences into vectors
        :param text: a list of sentences
        :param model: the model used to transform the sentence
        :return:
        """
        vectors = []
        for line in text:
            vectors.append(self.__transform(line, model))
        return vectors

    def __build_index(self, dimension):
        """
        function for building the index file,
        Do not use this function directly
        :param dimension: the dimension of the index file
        :return: an index file
        """
        self.base_index = faiss.IndexFlatL2(dimension)
        self.dimension = dimension
        return self.base_index, self.dimension

    def __check_model_name(self, model_name):
        if model_name is None:
            model = "bert-base-uncased"
        else:
            model = model_name
        return model

    def build_index(self, text:list, **kwargs):
        """
        function for building the index file
        :param text: original text list
        :param kwargs: model_name is an option, it is used to tell which model you want the transform function use
        :return:
        """
        model_name = kwargs.get("model_name")
        model = self.__check_model_name(model_name)

        vectors = self.__transform(text, model)
        dimension = vectors[0].shape[1]
        index, _ = self.__build_index(dimension)

        for vector in vectors:
            index.add(vector)

        return self.base_index

    def __build_index_with_id(self, dimension):
        """
        TODO: rewrite the index and the code of this function. Or maybe reconstruct the ID functions 6/6/2025
        function for building the index with ID file,
        Do not use this function directly
        :param dimension: the dimension of the index file
        :return: an index with id file
        """
        base_index, _ = self.__build_index(dimension)
        index = faiss.IndexIDMap(base_index)
        self.set_index(index)
        return self.index, self.dimension

    def build_index_with_id(self, text:list, identity:list, **kwargs):
        """
        TODO: rewrite the index and the code of this function. Or maybe reconstruct the ID functions 6/6/2025
        :param text:
        :param identity:
        :param kwargs:
        :return:
        """
        model_name = kwargs.get("model_name")
        model = self.__check_model_name(model_name)

        vectors = self.__transform(text, model)
        if len(vectors) != len(identity):
            raise ValueError(f"input text length {len(vectors)} does not equals to identity length {len(identity)}")

        dimension = vectors[0].shape[1]
        index, _ = self.__build_index_with_id(dimension)
        for i in range(0, len(vectors)):
            index.add_with_id(vectors[i], identity[i])

        return self.index

    def __add_example(self, vector):
        """
        function for add one vector to the index
        beware that the dimension of the vector must be the same as the index's, or it may raise error
        :param vector: the sample vector
        :return: None
        """
        if vector.shape[1] != self.dimension:
            raise ValueError(f"input text dimension does not match the index dimension\n{vector.shape[1]} != {self.dimension}")
        else:
            self.base_index.add(vector)

    def add_examples(self, text:list, **kwargs):
        """
        function for adding vectors into the index
        this function will transform the text into vectors
        :param text: the list of text
        :param kwargs: model_name is an option, if given, the transform function will use the given model
        :return: None
        """
        model_name = kwargs.get("model_name")
        model = self.__check_model_name(model_name)

        vectors = self.transform_word_to_vector(text, model)

        for vector in vectors:
            self.__add_example(vector)

    def save_index(self, filepath):
        """
        save the index file to your machine
        :param filepath: the filepath of the "index.faiss" file
        :return:
        """
        faiss.write_index(self.base_index, filepath)

    def read_index(self, filepath):
        """
        read the index file from your machine
        :param filepath: the filepath of the "index.faiss" file
        :return: None
        """
        self.base_index = faiss.read_index(filepath)

    def add_example_with_id(self, text:list, **kwargs):
        """
        TODO: needs to be reconstructed, something wrong with the structure 6/6/2025
        :param text:
        :param kwargs:
        :return:
        """
        return None

    def retrieve(self, text:str, top_k=3, **kwargs):
        """
        The function for retrieve the result.
        TODO: due to the ID problem, this part needs to be rewritten.
        TODO: this function needs to be tested,
        :param text:
        :param top_k:
        :param kwargs:
        :return:
        """
        model_name = kwargs.get("model_name")
        model = self.__check_model_name(model_name)

        vector = self.__transform(text, model)
        distances, ids = self.base_index.search(vector, top_k)
        return distances, ids