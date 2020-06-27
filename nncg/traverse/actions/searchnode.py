from copy import copy
from typing import List, Type
from nncg.traverse.traverseaction import TraverseAction
from nncg.traverse.tree import TreeNode, Edge


class SearchNode(TraverseAction):
    """
    Action for searching a node. This base action searches for a specific instant (comparing objects for equality).
    """
    result: List[List[TreeNode]]  # Stores the result. A single result is a List of TreeNodes which stores the complete
                                  # path that is visited to reach the result.
    cur_path_stack: List[TreeNode]

    def __init__(self, search_for):
        """
        Init the arction.
        :param search_for: The node to be searched for.
        """
        super().__init__()
        self.result = []
        self.search_for = search_for
        self.cur_path_stack = []

    def _pre_action(self, edge: Edge) -> bool:
        """
        Sets up a stack of passed nodes. See overwritten method for details.
        :param edge: Currently visited edge.
        :return: Always True.
        """
        self.cur_path_stack.append(edge.target)
        return True

    def _post_action(self, edge: Edge):
        """
        If target of currently visited Edge is equal to the searched node the stack is added to result.
        See overwritten method for details.
        :param edge: Currently visited edge.
        :return: Always True.
        """
        if edge.target == self.search_for:
            self.result.append(copy(self.cur_path_stack))
        self.cur_path_stack.pop()


class SearchNodeByType(SearchNode):
    """
    Variant of SearchNode. Here we search for a type of node.
    """
    def _post_action(self, edge: Edge):
        """
        If type of target of currently visited Edge is equal to the searched type the stack is added to result.
        See overwritten method for details.
        :param edge: Currently visited edge.
        :return: Always True.
        """
        if type(edge.target) == self.search_for:
            self.result.append(copy(self.cur_path_stack))
        self.cur_path_stack.pop()


class SearchNodeByName(SearchNode):
    """
    Variant of SearchNode. Here we search for a node name.
    """
    def _post_action(self, edge: Edge):
        """
        If the name of the target of the currently visited Edge is equal to the searched type the stack
        is added to result. See overwritten method for details.
        :param edge: Currently visited edge.
        :return: Always True.
        """
        if str(edge.target) == self.search_for:
            self.result.append(copy(self.cur_path_stack))
        self.cur_path_stack.pop()