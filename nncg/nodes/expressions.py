from __future__ import annotations
from typing import List, Dict, Optional
import numpy as np
from nncg.tools import _len
from nncg.traverse.tree import TreeNode


class Expression(TreeNode):
    """
    Node to express a general expression. It is usually used as part of Arithmetic oder meta
    nodes.
    """
    snippet = ''

    def __init__(self, snippet, **kwargs):
        """
        Init the node.
        :param snippet: The C code snippet with {} to be replace by further Expressions or Variables. E.g.
                        {var} / {stride0}
        :param kwargs:  A dictionary providing the data required for the above snippet. In the example above it would
                        be {'var': <Variable>, 'stride0': 1} for example.
        """
        super().__init__()
        self.snippet = snippet
        for a in kwargs:
            self.add_edge(a, kwargs[a], 'm_expr')

    def __str__(self):
        """
        Returns this Expression as a string.
        :return: The string.
        """
        return self.snippet.format(**self.edges)


class Constant(TreeNode):
    """
    Simple node just representing a constant.
    """
    def __init__(self, c):
        """
        Init this node.
        :param c: Arbitrary data that can return a string of itself.
        """
        super().__init__()
        self.c = c

    def __str__(self):
        """
        Get this Constant as a string.
        :return: The string.
        """
        return str(self.c)


class Variable(TreeNode):
    """
    Node representing a Variable. It can be a scalar value like (in C notation) float or int but also an array.
    In case of an array this Node will return the name of the array variable without indices. In the following
    we assume as an example that we want an array "float matrix[3][3]".
    For arrays a padding can be set. These increase the size of the array on declaration but does not affect
    this Variable elsewhere. The purpose is to enable different
    """
    type: str
    name: str
    dim: List
    alignment: str
    pads: List[List[int]] = None
    init_data: None

    def __init__(self,
                 type: str,
                 name: str,
                 dim: Optional[List[int]],
                 alignment: int,
                 index, init_data=None):
        """
        Init the Variable.
        :param type: Type as string, e.g. "float", "int" etc.
        :param name: Name of Variable. Index will be added as a number to get a unique name.
        :param dim: Dimensions in case of an array. None if no array.
        :param alignment: Desired alignment in bytes. Can be changed later. 0 means no alignment required.
        :param index: Number to get a unique name.
        :param init_data: Initial data. Can later be written into the C file.
        """
        super().__init__()
        self.decl_written = False
        self.index = index
        self.type = type
        self.name = name
        self.dim = dim
        self.set_alignment(alignment)
        self.init_data = init_data
        self.pads = _len(dim) * [[0, 0]]
        self.temporal_value = None

    def __str__(self):
        """
        Get name of Variable (including unique number).
        :return: The string.
        """
        if self.temporal_value is not None:
            return str(self.temporal_value)
        return '{name}_{index}'.format(name=self.name, index=self.index)

    def change_padding(self, pads: List[List[int]]):
        """
        Set a different padding size.
        :param pads: New padding.
        :return: None.
        """
        assert len(pads) == _len(self.dim)
        self.pads = pads

    def get_cast(self):
        """
        Get the string to cast something to the type of this variable.
        :return: The cast string.
        """
        return '({}*)'.format(self.type)

    def _get_dim_str(self):
        """
        Get the string for defining an array.
        :return: The string.
        """
        return ''.join(['[' + str(i + j[0] + j[1]) + ']' for i, j in zip(np.atleast_1d(self.dim), self.pads)])

    def get_def(self, write_init_data=True):
        """
        Get the string to define this Variable. Primarily useful for CHeaderNode.
        :param write_init_data: Should also the data be written into the C file for initialization?
        :return: The string.
        """
        if self.decl_written:
            return
        self.dim_str = self._get_dim_str()
        if self.init_data is not None and write_init_data:
            self.data_str = ','.join([np.format_float_scientific(f, precision=15) for f in (self.init_data.flatten())])
        else:
            self.data_str = '0'
        return 'static {type} {name}_{index} {alignment} {dim_str} = {{ {data_str} }};\n'.format(**self.__dict__)

    def get_pointer_decl(self):
        """
        This returns a string to declare this Variable as a pointer.
        :return: The declaration.
        """
        return '{type} *{name}_{index} {alignment};\n'.format(**self.__dict__)

    def set_alignment(self, bytes):
        """
        Set a new alignment.
        :param bytes: Address must be dividable by this number. 0 for no alignment.
        :return: None.
        """
        if bytes > 0:
            self.alignment = 'alignas({})'.format(8 * bytes)
        else:
            self.alignment = ''


class IndexedVariable(TreeNode):
    """
    This extension to a variable adds array indices to it ("[]").
    """
    def __init__(self, var, padding_to_offset=True):
        '''
        Init this IndexVariable.
        :param var: The Variable to add indices.
        :param padding_to_offset: If this is True, the padding will be bypassed by adding an offset to
                                  all accesses. Useful if a Variable later needs padding but this layer not so
                                  the padding is already added but bypassed here.
        '''
        super().__init__()
        self.add_edge('var', var)
        self.padding_to_offset = padding_to_offset

    def set_indices(self, indices: List[TreeNode]):
        """
        Set new indices.
        :param indices: List of indices, usually Variables, Expressions, etc.
        :return: None.
        """
        for i, idx in zip(indices, range(len(indices))):
            self.add_edge(str(idx), i, n_type='index')

    def __str__(self):
        """
        Get the string with Variable and indices.
        :return: The string.
        """
        s = str(self.get_node('var'))
        n = self.get_node_by_type('index')
        for i in n:
            s += '[' + str(i)
            if self.padding_to_offset:
                s += ' + ' + str(self.get_node('var').pads[n.index(i)][0])
            s += ']'
        return s