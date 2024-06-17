"""
6.101 Lab:
LISP Interpreter Part 2
"""

#!/usr/bin/env python3
import sys

sys.setrecursionlimit(20_000)

# NO ADDITIONAL IMPORTS!


class Frame:
    """
    class that acts as a frame in python does
    """

    def __init__(self, parent_f=None):
        """
        initialize frame with empty dictionary of bindings
        and it's parent frame
        """
        self.bindings = {}
        self.parent_frame = parent_f

    def set_val(self, var, expr):
        """
        creates a binding between an object name and the object
        """
        self.bindings[var] = expr

    def get_val(self, object_name):
        """
        Attempts to get value in this frame but if it isn't here
        it will try to get it from it's parent frame
        if it's not there also it will raise an error
        """
        if object_name in self.bindings:
            return self.bindings[object_name]
        else:
            if self.parent_frame is None:
                # print("rose from here1")
                raise SchemeNameError
            try:
                return self.parent_frame.get_val(object_name)
            except SchemeNameError as e:
                # print("rose from here2")
                raise SchemeNameError(e) from e

    def get_parent(self):
        """
        returns the parent frame
        """
        return self.parent_frame

    def __repr__(self):
        return f"bindings{self.bindings}"


class Function:
    """
    class to represent user defined functions
    """

    def __init__(self, param_list, body, enclosing_frame):
        """
        class initializer
        """
        self.param_num = len(param_list)
        self.params = {}
        for var_name in param_list:
            self.params[var_name] = None
        self.enclosing_frame = enclosing_frame  # frame that defined this function
        self.body = body

    def __call__(self, args, calling_frame):
        if len(args) != self.param_num:  # wrong number of arguments passed in
            raise SchemeEvaluationError

        for keys in self.params:
            try:
                self.params[keys] = args.pop(0)  # pop the first arg
            except SchemeEvaluationError as e:
                raise SchemeEvaluationError(e) from e

        # make new fame
        func_frame = Frame(self.enclosing_frame)
        for keys, vals in self.params.items():
            func_frame.set_val(keys, vals)

        # evaluate the body in the frame
        return evaluate(self.body, func_frame)


class Pair:
    """
    represents a cons cell
    """

    def __init__(self, car, cdr):
        self.car = car
        self.cdr = cdr

    def __str__(self):
        return f"Pair({self.car}, {self.cdr})"


#############################
# Scheme-related Exceptions #
#############################


class SchemeError(Exception):
    """
    A type of exception to be raised if there is an error with a Scheme
    program.  Should never be raised directly; rather, subclasses should be
    raised.
    """

    pass


class SchemeSyntaxError(SchemeError):
    """
    Exception to be raised when trying to evaluate a malformed expression.
    """

    pass


class SchemeNameError(SchemeError):
    """
    Exception to be raised when looking up a name that has not been defined.
    """

    pass


class SchemeEvaluationError(SchemeError):
    """
    Exception to be raised if there is an error during evaluation other than a
    SchemeNameError.
    """

    pass


############################
# Tokenization and Parsing #
############################


def number_or_symbol(value):
    """
    Helper function: given a string, convert it to an integer or a float if
    possible; otherwise, return the string itself

    >>> number_or_symbol('8')
    8
    >>> number_or_symbol('-5.32')
    -5.32
    >>> number_or_symbol('1.2.3.4')
    '1.2.3.4'
    >>> number_or_symbol('x')
    'x'
    """
    try:
        return int(value)
    except ValueError:
        try:
            return float(value)
        except ValueError:
            return value


def tokenize(source):
    """
    Splits an input string into meaningful tokens (left parens, right parens,
    other whitespace-separated values).  Returns a list of strings.

    Arguments:
        source (str): a string containing the source code of a Scheme
                      expression
    """
    tokens = []
    adj_chars = ""
    skip_line = False
    for c in source:
        if c in "() " or c == "\n":
            if c == "\n" and skip_line:
                skip_line = False

            if skip_line:  # skip if apart of skipped line
                continue

            if adj_chars:
                tokens.append(adj_chars)
                adj_chars = ""  # reset

            if c not in (" ", "\n"):
                tokens.append(c)  # add current c

        elif c == ";":  # skip line
            skip_line = True  # everytime you see \n make flag opposite
        else:
            if skip_line:  # skip if apart of skipped line
                continue
            adj_chars += c

    if adj_chars:
        tokens.append(adj_chars)
    return tokens


def parse(tokens):
    """
    Parses a list of tokens, constructing a representation where:
        * symbols are represented as Python strings
        * numbers are represented as Python ints or floats
        * S-expressions are represented as Python lists

    Arguments:
        tokens (list): a list of strings representing tokens
    """
    open_paren = 0

    def parse_expression(index):
        nonlocal open_paren
        val = tokens[index]
        if val not in ("(", ")"):  # it's a number or symbol
            return number_or_symbol(val), index + 1  # base case
        else:  # recursive case, seeing a s expression or (
            parsed = []
            if val == "(":
                open_paren += 1
            elif val == ")":
                open_paren -= 1
                if open_paren < 0:  # if more ) were seen before opening (
                    raise SchemeSyntaxError
            # open_paren+=1
            if index + 1 == len(
                tokens
            ):  # if next is last one return nothing, for () case
                return None, index + 1

            next_index = index + 1
            while next_index < len(tokens) and tokens[next_index] != ")":
                sub_expression, next_index = parse_expression(next_index)
                if sub_expression is not None:  # was just if sub_exp
                    parsed.append(sub_expression)

            if next_index < len(tokens) and tokens[next_index] == ")":
                open_paren -= 1
            return parsed, next_index + 1

    result, next_pos = parse_expression(0)
    if next_pos < len(tokens) or open_paren != 0:
        raise SchemeSyntaxError

    return result


######################
# Built-in Functions #
######################


def mult(args):
    """
    helper function to return the product of all entries in a list
    """
    if len(args) == 1:
        return args[0]
    else:
        return args[0] * mult(args[1:])


def div(args):
    """
    helper function to return quotion of all entries in a list
    """
    if len(args) == 1:
        return 1 / args[0]
    else:
        return args[0] / mult(args[1:])


def and_func(args, in_frame):
    """
    returns true if all true, false otherwise, does short circuiting
    """
    index = 0
    for i in args:
        if evaluate(i, in_frame) is False:
            return False
        index += 1
    return True


def or_func(args, in_frame):
    """
    returns true if one true, false if no trues, does short circuiting
    """
    for i in args:
        if evaluate(i, in_frame) is True or i == "#t":
            return True
    return False


def not_func(args, in_frame):
    """
    represents not boolean operator
    """
    if len(args) == 1:
        return not evaluate(args[0], in_frame)
    else:
        raise SchemeEvaluationError


def equal_func(args):
    """
    represents ==
    """
    first_num = args[0]
    for i in args:
        if i != first_num:
            return False
    return True


def cons(args, in_frame):
    if len(args) == 2:
        return Pair(evaluate(args[0], in_frame), evaluate(args[1], in_frame))
    else:
        raise SchemeEvaluationError


def get_car(cons_cell):
    """
    returns the car value
    """
    if len(cons_cell) == 1 and isinstance(cons_cell[0], Pair):
        return cons_cell[0].car
    else:
        raise SchemeEvaluationError


def get_cdr(cons_cell):
    """
    takes a list with the pair object in it
    returns the cdr value
    """
    if len(cons_cell) == 1 and isinstance(cons_cell[0], Pair):
        return cons_cell[0].cdr
    else:
        raise SchemeEvaluationError


def list_maker(args):
    """
    make a linked list with Pair objects
    """
    if len(args) == 0:
        return []
    else:
        return Pair(args[0], list_maker(args[1:]))


def is_list(obj=None):
    """
    taks a list object
    returns if obj is a list or not
    """
    if obj is None:  # nothing was passed in
        raise SchemeEvaluationError
    elif isinstance(obj, Pair) or obj == []:
        x = obj
        while True:
            try:
                x = get_cdr([x])
            except SchemeEvaluationError:
                # if x == []:  # if last cdr return was an empty list
                #     return True
                # else:
                #     return False
                return x==[]
    else:
        return False


def list_length(args):
    """
    takes list from the else in evaluate function and uses the first index
    gets lenght if it is a list
    """
    if len(args) != 1:
        raise SchemeEvaluationError
    if is_list(args[0]):
        length = 0
        x = [args[0]]
        no_exception = True
        while no_exception:
            try:
                x = [get_cdr(x)]
                length += 1
            except SchemeEvaluationError:
                no_exception = False
                return length  # went too far
    else:
        raise SchemeEvaluationError


def list_ref(args):
    """
    gets value at index
    """
    if len(args) != 2 or not isinstance(args[1], int):
        raise SchemeEvaluationError

    list_type = is_list(args[0])
    if (not list_type) and isinstance(args[0], Pair):  # not a list but a cons cell
        if args[1] == 0:
            return args[0].car
        else:
            raise SchemeEvaluationError
    elif list_type:
        copy_ind = args[1]
        curr_car = args[0]
        curr_cdr = args[0]
        while copy_ind >= 0:
            curr_car = get_car([curr_cdr])
            curr_cdr = get_cdr([curr_cdr])
            copy_ind -= 1
        return curr_car
    else:
        raise SchemeEvaluationError


def append_lists(args):
    """
    concatenates multipe lists together and returns result
    """
    if len(args) == 0:
        return []
    elif not is_list(args[0]):  # first arg must me a list
        raise SchemeEvaluationError
    else:  # add on to the ends of each list
        all_elements = []  # will hold all elements of all the lists
        for curr_list in args:
            if not (isinstance(curr_list, Pair) or curr_list == []):
                raise SchemeEvaluationError
            x = curr_list
            no_exception = True
            while no_exception:
                try:
                    all_elements.append(get_car([x]))
                    x = get_cdr([x])
                except SchemeEvaluationError:
                    no_exception = False
                    break
        return list_maker(all_elements)


def begin_func(args):
    """
    returns the last argument result
    """
    return args[len(args) - 1]


def evaluate_file(name_file, in_frame=None):
    """
    returns result of evaluating the expression in the file
    """
    file_obj = open(name_file)
    file_exp = str(file_obj.read())
    tokenized_version = tokenize(file_exp)
    parsed_version = parse(tokenized_version)
    return evaluate(parsed_version, in_frame)


def let_func(var_vals_body, in_frame):
    """
    deletes binding in current frame
    """
    new_frame = Frame(in_frame)
    for var_val in var_vals_body[0]:  # assign all vars in the new frame
        new_frame.set_val(var_val[0], evaluate(var_val[1], in_frame))
    return evaluate(var_vals_body[1], new_frame)


def set_func(args, in_frame):
    curr_frame = in_frame
    val = evaluate(args[1], in_frame)
    while curr_frame is not None:
        if args[0] in curr_frame.bindings:
            curr_frame.set_val(args[0], val)
            return val
        else:
            curr_frame = curr_frame.get_parent()
    raise SchemeNameError("from set")  # never found binding

def greater_sign(args):
    """
    greater than
    """
    for i in range(1, len(args)):
        if args[i]>=args[i-1]:
            return False
    return True

def lesser_sign(args):
    """
    greater than
    """
    for i in range(1, len(args)):
        if args[i]<=args[i-1]:
            return False
    return True

def greater_equal_sign(args):
    """
    greater than
    """
    for i in range(1, len(args)):
        if args[i]>args[i-1]:
            return False
    return True

scheme_builtins = {
    "+": sum,
    "-": lambda args: -args[0] if len(args) == 1 else (args[0] - sum(args[1:])),
    "*": mult,
    "/": div,

    "equal?": equal_func,
    
    ">": greater_sign,

    ">=": greater_equal_sign,

    "<": lesser_sign,

    "<=": lambda args: all(
        [args[i] >= args[i - 1] for i in range(1, len(args))]
    ),
    "and": and_func,
    "or": or_func,
    "not": not_func,
    "car": get_car,
    "cdr": get_cdr,
    "list": list_maker,
    "cons": cons,
    "list?": is_list,
    "length": list_length,
    "list-ref": list_ref,
    "append": append_lists,
    "begin": begin_func,
}


##############
# Evaluation #
##############


def evaluate(tree, curr_frame=None):
    """
    Evaluate the given syntax tree according to the rules of the Scheme
    language.

    Arguments:
        tree (type varies): a fully parsed expression, as the output from the
                            parse function
    """
    if curr_frame is None:  # if no frame was passed in
        curr_frame = make_initial_frame()

    if isinstance(tree, str):  # acts like a variable look up
        if tree == "#t":
            return True
        elif tree == "#f":
            return False
        else:
            return curr_frame.get_val(tree)
    elif isinstance(tree, int):
        return tree
    elif isinstance(tree, float):
        return tree
    elif isinstance(tree, list):  # list, recursive call
        if len(tree) == 0:  # represents an empty list
            return []
        elif tree[0] == "define":
            if isinstance(tree[1], list):
                curr_func = Function(tree[1][1:], tree[2], curr_frame)
                curr_frame.set_val(tree[1][0], curr_func)
                return curr_frame.get_val(tree[1][0])  # return value that was stored
            else:
                curr_frame.set_val(tree[1], evaluate(tree[2], curr_frame))
                return curr_frame.get_val(tree[1])  # return value that was stored
        elif tree[0] == "lambda":
            return Function(tree[1], tree[2], curr_frame)
        elif (
            tree[0] == "if"
        ):  # should always evaulate index 1 so shouldn't be this many branches
            if evaluate(tree[1], curr_frame):
                return evaluate(tree[2], curr_frame)
            else:
                return evaluate(tree[3], curr_frame)
        elif tree[0] in ("and", "or", "not"):
            return evaluate(tree[0], curr_frame)(tree[1:], curr_frame)
        elif tree[0] == "cons":
            if len(tree[1:]) == 2:
                return Pair(
                    evaluate(tree[1], curr_frame), evaluate(tree[2], curr_frame)
                )
            else:
                raise SchemeEvaluationError
        elif tree[0] == "del":
            if tree[1] in curr_frame.bindings:
                return curr_frame.bindings.pop(tree[1])
            else:
                raise SchemeNameError
        elif tree[0] == "list?":
            if len(tree) == 1 or len(tree) > 2:  # too little or too much arguments
                raise SchemeEvaluationError
            else:
                return is_list(evaluate(tree[1], curr_frame))
        elif tree[0] == "let":
            return let_func(tree[1:], curr_frame)
        elif tree[0] == "set!":
            ret_val = set_func(tree[1:], curr_frame)
            return ret_val
        else:
            updated_tree = [evaluate(i, curr_frame) for i in tree]
            if callable(updated_tree[0]):
                if updated_tree[0] in scheme_builtins.values() and tree[0] != "let":
                    return updated_tree[0](updated_tree[1:])
                else:
                    return updated_tree[0](updated_tree[1:], curr_frame)
            else:
                raise SchemeEvaluationError


def make_initial_frame():
    """
    makes an empty frame with parent that has built-ins
    """
    built_in_frame = Frame()  # frame with scheme_builitins
    built_in_frame.bindings = scheme_builtins
    return Frame(built_in_frame)


# THE PREVIOUS LAB, WHICH SHOULD BE THE STARTING POINT FOR THIS LAB.


if __name__ == "__main__":
    # NOTE THERE HAVE BEEN CHANGES TO THE REPL, KEEP THIS CODE BLOCK AS WELL
    # code in this block will only be executed if lab.py is the main file being
    # run (not when this module is imported)

    # 7) command line arguments
    global_frame = make_initial_frame()
    for file_name in sys.argv[1:]:
        evaluate_file(file_name, global_frame)

    import os

    sys.path.insert(0, os.path.dirname(os.path.realpath(__file__)))
    import schemerepl

    schemerepl.SchemeREPL(
        sys.modules[__name__], use_frames=True, verbose=False, global_frame=global_frame
    ).cmdloop()
    pass
