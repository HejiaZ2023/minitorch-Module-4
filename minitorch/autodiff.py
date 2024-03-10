from dataclasses import dataclass
from typing import Any, Dict, Iterable, Tuple

from typing_extensions import Protocol

# ## Task 1.1
# Central Difference calculation


def central_difference(f: Any, *vals: Any, arg: int = 0, epsilon: float = 1e-6) -> Any:
    r"""
    Computes an approximation to the derivative of `f` with respect to one arg.

    See :doc:`derivative` or https://en.wikipedia.org/wiki/Finite_difference for more details.

    Args:
        f : arbitrary function from n-scalar args to one value
        *vals : n-float values $x_0 \ldots x_{n-1}$
        arg : the number $i$ of the arg to compute the derivative
        epsilon : a small constant

    Returns:
        An approximation of $f'_i(x_0, \ldots, x_{n-1})$
    """

    vals0 = [(vals[i] - epsilon if i == arg else vals[i]) for i in range(len(vals))]
    vals1 = [(vals[i] + epsilon if i == arg else vals[i]) for i in range(len(vals))]
    return (f(*vals1) - f(*vals0)) / (2 * epsilon)
    # raise NotImplementedError("Need to implement for Task 1.1")


variable_count = 1


class Variable(Protocol):
    def accumulate_derivative(self, x: Any) -> None:
        pass

    @property
    def unique_id(self) -> int:
        pass

    def is_leaf(self) -> bool:
        pass

    def is_constant(self) -> bool:
        pass

    @property
    def parents(self) -> Iterable["Variable"]:
        pass

    def chain_rule(self, d_output: Any) -> Iterable[Tuple["Variable", Any]]:
        pass


def topological_sort(variable: Variable) -> Iterable[Variable]:
    """
    Computes the topological order of the computation graph.

    Args:
        variable: The right-most variable

    Returns:
        Non-constant Variables in topological order starting from the right.
    """
    d: Dict[int, int] = dict()
    var_id_dict: Dict[int, Variable] = dict()

    def process_var(var: Variable, d: Dict[int, int], iters: int) -> None:
        d[var.unique_id] = iters
        var_id_dict[var.unique_id] = var
        assert iters < len(d)
        iters += 1
        if var.is_constant():
            return
        # assert var.history is not None, f"type: {type(var)}, tuple: {var.tuple()}"
        for p in var.parents:
            if (p.unique_id in d) and (d[p.unique_id] > iters):
                continue
            process_var(p, d, iters)

    process_var(variable, d, 0)
    ret = sorted(d.items(), key=lambda item: item[1])
    return [var_id_dict[p[0]] for p in ret]
    # raise NotImplementedError("Need to implement for Task 1.4")


def backpropagate(variable: Variable, deriv: Any) -> None:
    """
    Runs backpropagation on the computation graph in order to
    compute derivatives for the leave nodes.

    Args:
        variable: The right-most variable
        deriv  : Its derivative that we want to propagate backward to the leaves.

    No return. Should write to its results to the derivative values of each leaf through `accumulate_derivative`.
    """
    var_list = topological_sort(variable)
    var_id_dict: Dict[int, Variable] = dict()
    deriv_dict: Dict[int, Any] = dict()
    for var in var_list:
        var_id_dict[var.unique_id] = var
        deriv_dict[var.unique_id] = 0
    deriv_dict[variable.unique_id] += deriv
    for var in var_list:
        if var.is_leaf():
            var.accumulate_derivative(deriv_dict[var.unique_id])
            continue
        if var.is_constant():
            continue
        t = var.chain_rule(deriv_dict[var.unique_id])
        for parent, parent_deriv in t:
            deriv_dict[parent.unique_id] += parent_deriv
    # raise NotImplementedError("Need to implement for Task 1.4")


@dataclass
class Context:
    """
    Context class is used by `Function` to store information during the forward pass.
    """

    no_grad: bool = False
    saved_values: Tuple[Any, ...] = ()

    def save_for_backward(self, *values: Any) -> None:
        "Store the given `values` if they need to be used during backpropagation."
        if self.no_grad:
            return
        self.saved_values = values

    @property
    def saved_tensors(self) -> Tuple[Any, ...]:
        return self.saved_values
