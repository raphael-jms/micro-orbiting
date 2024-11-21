import re
import sympy as sp
import numpy as np

def split_expression(expr, symbol):
    """
    Split an expression by a symbol. The symbol must not be within brackets.
    The symbol should be + or *.
    """
    parts = []
    current_part = ""
    bracket_count = 0

    for char in expr:
        if char == "(":
            bracket_count += 1
        elif char == ")":
            bracket_count -= 1

        if char == symbol and bracket_count == 0:
            parts.append(current_part.strip())
            current_part = ""
        else:
            current_part += char

    if current_part:
        parts.append(current_part.strip())

    return {"sym":symbol, "parts":parts}

def split_recursively(expr, max_length=500, level=0, temp_var_prefix="temp"):
    """
    Split an expression recursively so that the parts are not longer than max_length.
    Returns a tree structure containing nested list with the single parts and the 
    mathematical operation/symbol that should be used to combine them.
    """
    # print(f"{level*'  '}level: {level}")
    # print(f"{level*'  '}{expr}")
    for i, part in enumerate(expr["parts"]):
        # print(f"{level*'  '}i: {i}")
        # print(f"{level*'  '}part: {part}")
        if len(part) > max_length:
            split_result = split_expression(part, "+")
            if split_result["parts"][0] == part:
                split_result = split_expression(part, "*")

                # Check if the whole expression is enclosed by brackets and remove if necessary
                for j, split_result_part in enumerate(split_result["parts"]):
                    if split_result_part[0] == "(" and split_result_part[-1] == ")":
                        split_result["parts"][j] = split_result_part[1:-1]

                if split_result["parts"][0] == part:
                    print(f"Could not further split the expr: {part}")
                    continue
            expr["parts"][i] = split_recursively(split_result, max_length=max_length, level=level+1)

    return expr

def gen_code(tree, name="temp"):
    """
    Takes the tree structure from split_recursively and generates the actual code. Done by 
    adding temporary variables that store the intermediate results of the expression and
    then combine them at the end.
    """
    code = ""
    symbols = []
    for i, part in enumerate(tree["parts"]):
        if isinstance(part, dict):
            new_code, new_name = gen_code(part, f"{name}_{i}")
            symbols.append(new_name)
            code += new_code
        else:
            symbols.append(part)

    code += name + " = " + symbols[0]
    for s in symbols[1:]:
        code += tree["sym"] + " " + s
    code += "\n"

    return code, name
    
def handle_final_expression(expression, max_length=80, expr_name="expr"):
    # Remove any whitespace from the expression
    expression = re.sub(r'\s+', '', expression)

    # do not split up e.g. r**2
    expression = expression.replace("**", "ThisIsAPower")
    result = split_recursively({"sym":"", "parts":[expression]},max_length=max_length)
    code, _ = gen_code(result["parts"][0], expr_name)

    return code.replace("ThisIsAPower", "**")

def handle_matrix(matrix, symbol="matr"):
    code = ""
    nrows, ncols = matrix.shape
    for i in range(nrows):
        for j in range(ncols):
            code += f"{symbol}_{i}_{j} = {matrix[i,j]}\n"

    code += f"{symbol} = np.zeros(({nrows}, {ncols}))\n"
    for i in range(nrows):
        for j in range(ncols):
            code += f"{symbol}[{i}, {j}] = {symbol}_{i}_{j}\n"

    return code

def python_code_from_sympy(nonlin_expr, cost_mat, variables_tuple, code_file="cost_function_v2.py"):
    """
    Theoretically, sp.python gives us valid python code. However, the expression is way too long
    for the Python interpreter to handle. We need to split the expression into smaller parts
    which is done in the rest of the code
    """
    python_code = sp.python(nonlin_expr)

    # Search each line of python_code and check if it starts with "e =" (the final expression)
    lines_before = ""
    final_expression = ""
    lines_after = ""
    list_of_symbols = []
    split = python_code.split('\\n')
    if split[0] == python_code:
        split = python_code.split('\n')

    for line in split:
        if line.startswith('e ='):
            final_expression = line.replace("e = ", "")
        elif final_expression  == "":
            if "Symbol" in line:
                list_of_symbols.append(re.search(r"Symbol\('(.+?)'\)", line).group(1))
            # else:
            #     lines_before += line + "\n"
            if not "Symbol('omega_theta')" in line and not "Symbol('r')" in line:
                lines_before += line + "\n"
        else:
            lines_after += line + "\n"

    expression_name = "P"
    code = handle_final_expression(final_expression, expr_name=expression_name)
    matrix_name = "cost_matrix"
    code += handle_matrix(cost_mat, symbol=matrix_name)

    # Add the symbols to the beginning of the code again just to be sure
    lines_before += "\n"
    lines_before += "c0_1 = Symbol('c0_1')" + "\n"
    lines_before += "c0_2 = Symbol('c0_2')" + "\n"
    lines_before += "c0_3 = Symbol('c0_3')" + "\n"
    lines_before += "c0_4 = Symbol('c0_4')" + "\n"
    lines_before += "c0_5 = Symbol('c0_5')" + "\n"

    python_code = lines_before + code + "\n" + lines_after
    python_code = python_code.replace("\n", "\n    ") # add indentation
    # add the lambdify part
    python_code = "import sympy as sp\n"+\
        "from sympy import Symbol, Float, Max, sqrt\n"+\
        "import numpy as np\n\n"+\
        f"def get_cost_function(r, omega_theta):\n    " +\
        python_code + \
        f"\n    nonlinear_cost = sp.lambdify({variables_tuple}, {expression_name})\n" +\
        f"    quadratic_cost = {matrix_name}\n\n" +\
        "    def cost_function(x):\n" +\
        "        return nonlinear_cost(x[0], x[1], x[2], x[3], x[4]) + x.T @ quadratic_cost @ x\n" +\
        "    return cost_function\n"

    # Write to file
    with open(code_file, "w") as file:
        file.write(python_code)

    return python_code


if __name__ == "__main__":
    # python_code=""python_code
    # with open("./tests/parse/code.txt", "r") as file:
    #     python_code = file.read()

    # x0_1 = sp.Symbol('x0_1')
    # x0_2 = sp.Symbol('x0_2')
    # x0_3 = sp.Symbol('x0_3')
    # x0_4 = sp.Symbol('x0_4')
    # x0_5 = sp.Symbol('x0_5')
    # x0_6 = sp.Symbol('x0_6')
    # r = sp.Symbol('r')
    # omega_theta = sp.Symbol('omega_theta')
    # var_tuple = (x0_1, x0_2, x0_3, x0_4, x0_5, x0_6)

    # code = handle_final_expression(python_code, 120)
    # print(code)


    # python_code = "x0_1*(Float('3.0352733686067035', precision=53)*x0_1 - Float('0.26158209074875871', precision=53)*x0_2 - Float('0.34220979637646676', precision=53)*x0_3 + Float('0.096296296296297657', precision=53)*x0_4 + Float('0.06515752765752747', precision=53)*x0_5 + Float('0.07002765752765705', precision=53)*x0_6) + x0_2*(-Float('0.2615820907487576', precision=53)*x0_1 + Float('4.757922478755809', precision=53)*x0_2 - Float('1.0013668430335088', precision=53)*x0_3 + Float('0.042081529581529303', precision=53)*x0_4 + Float('0.14452861952861895', precision=53)*x0_5 + Float('0.10836339586339738', precision=53)*x0_6) + x0_3*(-Float('0.34220979637646409', precision=53)*x0_1 - Float('1.0013668430335123', precision=53)*x0_2 + Float('5.767121612954945', precision=53)*x0_3 + Float('0.019029581529581119', precision=53)*x0_4 + Float('0.050895863395864049', precision=53)*x0_5 + Float('0.15361952861952943', precision=53)*x0_6) + x0_4*(Float('0.096296296296297768', precision=53)*x0_1 + Float('0.042081529581529525', precision=53)*x0_2 + Float('0.019029581529581008', precision=53)*x0_3 + Float('0.34303350970017638', precision=53)*x0_4 + Float('0.056267035433702206', precision=53)*x0_5 + Float('0.058106862273529014', precision=53)*x0_6) + x0_5*(Float('0.06515752765752747', precision=53)*x0_1 + Float('0.14452861952862062', precision=53)*x0_2 + Float('0.05089586339586516', precision=53)*x0_3 + Float('0.056267035433702116', precision=53)*x0_4 + Float('0.39139610389610402', precision=53)*x0_5 + Float('0.11291887125220515', precision=53)*x0_6) + x0_6*(Float('0.070027657527656717', precision=53)*x0_1 + Float('0.10836339586339738', precision=53)*x0_2 + Float('0.15361952861952854', precision=53)*x0_3 + Float('0.058106862273529118', precision=53)*x0_4 + Float('0.11291887125220437', precision=53)*x0_5 + Float('0.46098484848484877', precision=53)*x0_6) + (Float('0.056280619243582519', precision=53)*omega_theta**2*r**2*x0_1**2 + Float('0.22381130529278637', precision=53)*omega_theta**2*r**2*x0_1*x0_2 - Float('0.58657830509682474', precision=53)*omega_theta**2*r**2*x0_1*x0_3 + Float('0.032921810699588383', precision=53)*omega_theta**2*r**2*x0_1*x0_4 + Float('0.058051413606969859', precision=53)*omega_theta**2*r**2*x0_1*x0_5 + Float('0.073635829191384311', precision=53)*omega_theta**2*r**2*x0_1*x0_6 + Float('0.24545276397128069', precision=53)*omega_theta**2*r**2*x0_2**2 - Float('1.0467568097197706', precision=53)*omega_theta**2*r**2*x0_2*x0_3 + Float('0.067500400833734542', precision=53)*omega_theta**2*r**2*x0_2*x0_4 + Float('0.1289936401047489', precision=53)*omega_theta**2*r**2*x0_2*x0_5 + Float('0.23548714659825826', precision=53)*omega_theta**2*r**2*x0_2*x0_6 + Float('2.707790426308947', precision=53)*omega_theta**2*r**2*x0_3**2 - Float('0.16626583293250063', precision=53)*omega_theta**2*r**2*x0_3*x0_4 - Float('0.26840895729784664', precision=53)*omega_theta**2*r**2*x0_3*x0_5 - Float('0.16191545080434055', precision=53)*omega_theta**2*r**2*x0_3*x0_6 + Float('0.0048598863413678584', precision=53)*omega_theta**2*r**2*x0_4**2 + Float('0.017581458322198901', precision=53)*omega_theta**2*r**2*x0_4*x0_5 + Float('0.025373666114406668', precision=53)*omega_theta**2*r**2*x0_4*x0_6 + Float('0.016978426237685307', precision=53)*omega_theta**2*r**2*x0_5**2 + Float('0.06498138349990186', precision=53)*omega_theta**2*r**2*x0_5*x0_6 + Float('0.27022517948443875', precision=53)*omega_theta**2*r**2*x0_6**2 - Float('0.0062415864884999284', precision=53)*omega_theta*r**2*x0_1**3 - Float('0.039784358302879852', precision=53))"    # Test expression
    # python_code = "a + b + c + (d  +e + f) * (g + h + i) + j * (4 + 7 + k)"
    python_code = "a + b + c + (d  +e + f) * (g + h + i) + j * (4 + 7 + k) + Max(3, 4, 5)*(3, 1 + Min(3, z))"
    print(python_code)
    # print(handle_final_expression(python_code, 80))
    print(handle_final_expression(python_code, 2))
