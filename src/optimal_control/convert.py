import re


def ode_matlab_to_jax(code: str, state_width: int) -> str:
    # Replace state syntax
    code = re.sub(r"x\((\d+)\)", lambda match: f"x[{int(match.group(1))-1}]", code)

    # Replace derivative syntax
    code = re.sub(r"dx\(\s(\d+)\)", lambda match: f"dx[{int(match.group(1))-1}]", code)

    # Replace constant syntax
    code = re.sub(r"k\((\d+)\)", lambda match: f"k[{int(match.group(1))-1}]", code)

    # Replace comments
    code = re.sub(r"%", r"#", code)

    # Remove line endings
    code = re.sub(r";", r"", code)

    # Replace matlab math symbols with python equivalents
    code = re.sub(r"\^", r"**", code)

    # Add header
    code = f"dx = [None]*{state_width}\n" + code

    # Add footer
    code = code + "\nreturn dx"

    return code


def constants_matlab_to_jax(code: str, num_constants: int) -> str:
    # Replace constant syntax
    code = re.sub(r"k\(\s+(\d+)\)", lambda match: f"k[{int(match.group(1))-1}]", code)

    # Replace x*10^y -> xey
    code = re.sub(r"(\d+)\*10\^(\d+)", r"\1e\2", code)

    # Replace comments
    code = re.sub(r"%", r"#", code)

    # Remove line endings
    code = re.sub(r";", r"", code)

    # Add constant list & stacking
    code = f"k = [None]*{num_constants}\n\n" + code

    return code
