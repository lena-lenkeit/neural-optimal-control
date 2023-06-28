"""Utilities for loading ODE models defined in SBML into jax-compatible constructs"""

from typing import Any, Callable, Dict

import libsbml


def species_to_dict(species_list: libsbml.ListOfSpecies) -> Dict[str, float]:
    species_dict = {}
    for i in range(len(species_list)):
        species: libsbml.Species = species_list[i]
        species_dict[species.getIdAttribute()] = species.getInitialConcentration()

    return species_dict


def parameters_to_dict(parameter_list: libsbml.ListOfParameters) -> Dict[str, float]:
    parameter_dict = {}
    for i in range(len(parameter_list)):
        parameter: libsbml.Parameter = parameter_list[i]
        parameter_dict[parameter.getIdAttribute()] = parameter.getValue()

    return parameter_dict


def species_references_to_dict(
    species_references_list: libsbml.ListOfSpeciesReferences,
) -> Dict[str, float]:
    species_reference_dict = {}
    for i in range(len(species_references_list)):
        species_reference: libsbml.SpeciesReference = species_references_list[i]
        species_reference_dict[
            species_reference.getSpecies()
        ] = species_reference.getStoichiometry()

    return species_reference_dict


def reactions_to_dict(
    reaction_list: libsbml.ListOfReactions,
) -> Dict[str, Dict[str, float]]:
    reaction_dict = {}
    for i in range(len(reaction_list)):
        reaction: libsbml.Reaction = reaction_list[i]
        kinetic_law: libsbml.KineticLaw = reaction.getKineticLaw()
        reaction_dict[reaction.getIdAttribute()] = {
            "parameters": parameters_to_dict(kinetic_law.getListOfParameters()),
            "reactants": species_references_to_dict(reaction.getListOfReactants()),
            "products": species_references_to_dict(reaction.getListOfProducts()),
        }

    return reaction_dict


def compartments_to_dict(
    compartment_list: libsbml.ListOfCompartments,
) -> Dict[str, float]:
    compartment_dict = {}
    compartment: libsbml.Compartment
    for compartment in compartment_list:
        compartment_dict[compartment.getIdAttribute()] = compartment.getSize()

    return compartment_dict


def function_definitions_to_dict(
    fn_def_list: libsbml.ListOfFunctionDefinitions,
) -> Dict[str, Any]:
    """Converts a list of function definitions into jax-compatible functions, annotated
    with their arguments"""

    fn_def_dict = {}
    fn_def: libsbml.FunctionDefinition
    for fn_def in fn_def_list:
        # Parse arguments
        arguments = []
        for i in range(fn_def.getNumArguments()):
            argument: libsbml.ASTNode = fn_def.getArgument(i)
            arguments.append(argument.getName())

        # Parse body
        body_fn = ast_to_lambda(fn_def.getBody())

        # Store
        fn_def_dict[fn_def.getIdAttribute()] = {"arguments": arguments, "body": body_fn}

    return fn_def_dict


def eval_ast_node(node: libsbml.ASTNode, context: Dict[str, Any]) -> float:
    """Evaluates the numerical value of a libsbml.ASTNode, given a context of other
    values and callable functions"""

    if node.isOperator():
        # Unary negation operator
        if node.getNumChildren() == 1:
            assert node.getType() == libsbml.AST_MINUS

            val = eval_ast_node(node.getChild(0), context)
            return -val

        # Binary operators
        elif node.getNumChildren() == 2:
            operators = {
                libsbml.AST_PLUS: lambda a, b: a + b,
                libsbml.AST_MINUS: lambda a, b: a - b,
                libsbml.AST_TIMES: lambda a, b: a * b,
                libsbml.AST_DIVIDE: lambda a, b: a / b,
                libsbml.AST_POWER: lambda a, b: a**b,
            }

            op = operators[node.getType()]
            left = eval_ast_node(node.getLeftChild(), context)
            right = eval_ast_node(node.getRightChild(), context)
            val = op(left, right)

            return val

        else:
            raise ValueError(node.getNumChildren())

    elif node.isFunction():
        if node.isUserFunction():
            # Evaluate all arguments
            fn_args = context[node.getName()]["arguments"]
            fn_argument_vals = {
                fn_args[i]: eval_ast_node(node.getChild(i), context)
                for i in range(node.getNumChildren())
            }

            # Call the function with the arguments as context
            fn_body = context[node.getName()]["body"]
            fn_ret_val = fn_body(context=fn_argument_vals)

            return fn_ret_val

        else:
            functions = {libsbml.AST_FUNCTION_POWER: lambda a, b: a**b}

            fn = functions[node.getType()]
            left = eval_ast_node(node.getLeftChild(), context)
            right = eval_ast_node(node.getRightChild(), context)
            val = fn(left, right)

            return val

    elif node.isNumber():
        return node.getReal()

    elif node.isName():
        return context[node.getName()]

    else:
        raise TypeError(
            "Unsupported AST node type:", node, node.getName(), node.getType()
        )


def ast_to_lambda(node: libsbml.ASTNode) -> Callable[[Dict[str, Any]], float]:
    """Turns a libsbml.ASTNode into a jax-compatible callable function"""

    def ast_fn(context: Dict[str, Any]):
        return eval_ast_node(node, context)

    return ast_fn


def model_to_lambda(model: libsbml.Model) -> Callable[[float, dict, dict], dict]:
    """Turns a libsbml.Model into a jax-/diffrax-compatible ODE Term"""

    # Load model-global context
    fn_defs = function_definitions_to_dict(model.getListOfFunctionDefinitions())
    global_parameters = parameters_to_dict(model.getListOfParameters())
    compartments = compartments_to_dict(model.getListOfCompartments())

    def ode_fn(t: float, y: dict, overrides: dict) -> dict:
        dy_dt = {k: 0.0 for k in y}

        # Iterate over reactions to obtain the full dy_dt
        reaction: libsbml.Reaction
        for reaction in model.getListOfReactions():
            kinetic_law: libsbml.KineticLaw = reaction.getKineticLaw()

            # Load reaction-local context
            local_parameters = parameters_to_dict(kinetic_law.getListOfParameters())

            # Prepare evaluation context
            context = {}

            context.update(fn_defs)
            context.update(global_parameters)
            context.update(local_parameters)
            context.update(compartments)
            context.update(y)

            # Update with overrides last, so that any value in the context could
            # potentially be modified
            context.update(overrides)

            # Apply reaction, taking into account product and reactant stochiometry to
            # update dy_dt accordingly
            # TODO: Compartment sizes
            reaction_fn = ast_to_lambda(kinetic_law.getMath())
            reaction_vel = reaction_fn(context=context)

            reactant: libsbml.SpeciesReference
            for reactant in reaction.getListOfReactants():
                dy_dt[reactant.getSpecies()] -= (
                    reactant.getStoichiometry() * reaction_vel
                )

            product: libsbml.SpeciesReference
            for product in reaction.getListOfProducts():
                dy_dt[product.getSpecies()] += product.getStoichiometry() * reaction_vel

        return dy_dt

    return ode_fn
