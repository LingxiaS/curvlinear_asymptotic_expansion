import streamlit as st
import sympy as sp
from typing import List, Union, Tuple

# --- SymPy Core Calculation Functions ---

def curvilinear_gradient(phi, n_vec, s_vec, epsilon, rho, s, K, order):
    """Calculates the curvilinear gradient (nabla phi)."""
    grad_rho = sp.expand(n_vec * (1/epsilon) * sp.diff(phi, rho))
    # Use .removeO() to get the explicit series expansion terms
    grad_s = sp.expand(s_vec * (1/(1 + epsilon*rho*K)).series(epsilon, 0, order).removeO() * sp.diff(phi, s))
    return sp.expand(grad_rho + grad_s)

def curvilinear_divergence(n_component, s_component, epsilon, rho, s, K):
    """Calculates the curvilinear divergence (div(V))."""
    # Assuming V = n_vec * n_component + s_vec * s_component
    term1 = sp.expand((1/epsilon) * sp.diff(n_component, rho))
    term2 = sp.expand(sp.diff(s_component, s))
    term3 = sp.expand(K * n_component)
    div = sp.collect(term1 + term2 + term3, epsilon)
    return div

def curvilinear_laplacian(phi, n_vec, s_vec, epsilon, rho, s, K, order):
    """Calculates the curvilinear Laplacian (nabla^2 phi)."""
    # 1. Calculate grad(phi)
    grad_phi = curvilinear_gradient(phi, n_vec, s_vec, epsilon, rho, s, K, order)
    
    # 2. Extract n and s components of grad(phi)
    n_comp_grad = grad_phi.subs({s_vec: 0, n_vec: 1}) 
    s_comp_grad = grad_phi.subs({s_vec: 1, n_vec: 0})
    
    # 3. Calculate divergence of the gradient (Laplacian)
    lap_phi = curvilinear_divergence(n_comp_grad, s_comp_grad, epsilon, rho, s, K)
    
    return sp.collect(lap_phi, epsilon)


# --- Streamlit Page Configuration ---
st.set_page_config(page_title="Curvilinear Asymptotic Expansion Tool", layout="wide")

st.title("Curvilinear Asymptotic Expansion Tool")
st.markdown("---")


# --- Main Instruction and Formula ---

st.header("Theory and Formula")
st.markdown("""
This tool performs asymptotic expansion for vector calculus operators in a curvilinear coordinate system $(\\rho, s)$, where $\\epsilon$ is the small parameter and $K(s)$ is the curvature.

The basis vectors are $\\mathbf{n}$ (normal) and $\\mathbf{s}$ (tangential).
""")

st.subheader("Curvilinear Gradient ($\nabla \\phi$):")
st.latex(r'''
\nabla \phi = \mathbf{n} \left(\frac{1}{\epsilon}\frac{\partial \phi}{\partial \rho}\right) + \mathbf{s} \left(\frac{1}{1 + \epsilon \rho K}\frac{\partial \phi}{\partial s}\right)
''')

st.subheader("Curvilinear Divergence ($\nabla \cdot \mathbf{V}$):")
st.latex(r'''
\nabla \cdot \mathbf{V} = \frac{1}{\epsilon} \frac{\partial V_n}{\partial \rho} + \frac{\partial V_s}{\partial s} + K V_n
''')
st.caption("Where $\\mathbf{V} = V_n \\mathbf{n} + V_s \\mathbf{s}$.")
st.markdown("---")


# --- Sidebar / Input Area ---
st.sidebar.header("Calculation Parameters")

# 1. Variable Name Symbol
variable_name = st.sidebar.text_input(
    "1. Variable Name (e.g., V, H):",
    value="H"
)

# 2. Variable Type
variable_type = st.sidebar.selectbox(
    "2. Variable Type:",
    ("Scalar ($\phi$)", "Vector ($\mathbf{V}$)")
)

# 3. Apply Which Expansion
if variable_type == "Scalar ($\phi$)":
    op_choices = ["Gradient ($\nabla$)", "Laplacian ($\nabla^2$)"]
else:
    op_choices = ["Divergence ($\nabla \cdot$)"]
    
operation = st.sidebar.selectbox(
    "3. Select Operator:",
    op_choices
)

# 4. Vector Components (if Vector type is chosen)
n_component_str = "0"
s_component_str = "0"
if variable_type == "Vector ($\mathbf{V}$)":
    st.sidebar.subheader("Vector Components")
    n_component_str = st.sidebar.text_input(
        f"Normal Component ($V_n$):",
        value="0"
    )
    s_component_str = st.sidebar.text_input(
        f"Tangential Component ($V_s$):",
        value="0"
    )

# 5. Expansion Order (for gradient/Laplacian which involve series)
order = st.sidebar.number_input(
    "4. Expansion Order (for series, e.g., 2):",
    min_value=0,
    value=2
)

# 6. Variable Expansion 
st.sidebar.markdown("---")
st.sidebar.subheader("Variable Expansion")

do_variable_expansion = st.sidebar.checkbox(
    f"5. Expand {variable_name} in $\\epsilon$", 
    True, 
    help="If checked, the variable is replaced by a user-defined series expansion."
)

expansion_terms: List[sp.Expr] = []
if do_variable_expansion:
    expansion_terms_area = st.sidebar.text_area(
        "Expansion Terms (comma-separated, $V_0, V_1, ...$):",
        value="K(s), -rho * K(s)**2",
        help="Terms $V_0, V_1, ...$ corresponding to $\\epsilon^0, \\epsilon^1, ...$"
    )
    try:
        expansion_terms = [sp.sympify(t.strip()) for t in expansion_terms_area.split(',')]
    except Exception:
        # Error will be caught during final execution
        pass 


# --- Core Execution Function (with UnhashableParamError fix) ---

@st.cache_data
def execute_calculation(
    op_type: str, 
    variable_name: str,
    v_type: str,
    n_comp_str: str,
    s_comp_str: str,
    _expansion_terms: List[sp.Expr], # FIX: Underscore to disable Streamlit hashing
    order: int
) -> Tuple[str, str]:
    
    # 1. Define base symbols
    rho, s, epsilon = sp.symbols('rho s epsilon')
    K = sp.Function('K')(s)
    n_vec, s_vec = sp.symbols('n_vec s_vec')

    # 2. Define the Variable (H or V)
    if v_type == "Scalar ($\phi$)":
        if _expansion_terms:
            phi = sp.sympify(0)
            for i, term in enumerate(_expansion_terms):
                phi += term * (epsilon ** i)
        else:
            phi = sp.Function(variable_name)(rho, s)
            
        target_expr = phi
        
    elif v_type == "Vector ($\mathbf{V}$)":
        Vn = sp.sympify(n_comp_str)
        Vs = sp.sympify(s_comp_str)
        target_expr = Vn * n_vec + Vs * s_vec
        
    else:
        raise ValueError("Invalid variable type.")

    # 3. Perform the calculation based on the selected operator
    if op_type == "Gradient ($\nabla$)" and v_type == "Scalar ($\phi$)":
        result_expr = curvilinear_gradient(target_expr, n_vec, s_vec, epsilon, rho, s, K, order)
        op_symbol = r"\nabla " + variable_name
        
    elif op_type == "Laplacian ($\nabla^2$)" and v_type == "Scalar ($\phi$)":
        result_expr = curvilinear_laplacian(target_expr, n_vec, s_vec, epsilon, rho, s, K, order)
        op_symbol = r"\nabla^2 " + variable_name
        
    elif op_type == "Divergence ($\nabla \cdot$)" and v_type == "Vector ($\mathbf{V}$)":
        result_expr = curvilinear_divergence(Vn, Vs, epsilon, rho, s, K)
        op_symbol = r"\nabla \cdot \mathbf{" + variable_name + "}"
        
    else:
        raise ValueError("Invalid operation/type combination.")

    # 4. Collect terms by epsilon power
    final_result = sp.collect(result_expr.subs(sp.Order(epsilon), 0), epsilon)

    return op_symbol, sp.latex(final_result)


# --- Execution and Display ---

st.subheader("Result")

if st.sidebar.button("Execute Calculation"):
    if (variable_type == "Vector ($\mathbf{V}$)" and not (n_component_str or s_component_str)):
         st.warning("Please define at least one vector component ($V_n$ or $V_s$).")
    elif not variable_name:
        st.warning("Please enter a Variable Name.")
    else:
        with st.spinner(f"Calculating {operation} of {variable_name}..."):
            try:
                # Call core calculation function (using expansion_terms_list for the cached argument)
                latex_result = execute_calculation(
                    operation, 
                    variable_name,
                    variable_type,
                    n_component_str if variable_type == "Vector ($\mathbf{V}$)" else "0",
                    s_component_str if variable_type == "Vector ($\mathbf{V}$)" else "0",
                    expansion_terms, # This maps to _expansion_terms
                    order
                )

                st.success(f"Calculation Successful for ${latex_result[0]}$")
                
                # Display result
                st.subheader(f"Expanded Form of ${latex_result[0]}$:")
                st.latex(latex_result[1])
                
            except Exception as e:
                st.error(f"Calculation Failed: Check your input expressions.")
                st.exception(e)

st.markdown("---")


# --- Example Demo Usage ---
st.header("Example: Laplacian of Expanded H")
st.markdown("""
This demo replicates a common use case where the variable $H$ is expanded to first order.

**Inputs used in this example:**
* **Variable Name:** H
* **Variable Type:** Scalar ($\phi$)
* **Operator:** Laplacian ($\nabla^2$)
* **Expansion Order:** 2
* **Expansion Terms:** `K(s)`, `-rho * K(s)**2` (i.e., $H_0 = K(s)$, $H_1 = -\rho K(s)^2$)
""")

# Example variables defined in the same way as the main app
ex_rho, ex_s, ex_epsilon = sp.symbols('rho s epsilon')
ex_K = sp.Function('K')(ex_s)

# Expansion: H0 + eps*H1
ex_H_0 = ex_K
ex_H_1 = -ex_rho * ex_K**2
ex_H_expanded = ex_H_0 + ex_H_1 * ex_epsilon + sp.Order(ex_epsilon**3)

ex_n_vec, ex_s_vec = sp.symbols('n_vec s_vec')

# Calculate the Laplacian of the expanded H
ex_lap_H = curvilinear_laplacian(ex_H_expanded, ex_n_vec, ex_s_vec, ex_epsilon, ex_rho, ex_s, ex_K, 2)
ex_lap_H_final = sp.collect(ex_lap_H.subs(sp.Order(ex_epsilon), 0), ex_epsilon)

st.subheader("Result for $\\nabla^2 H$:")
st.latex(sp.latex(ex_lap_H_final))

st.caption("Application powered by Streamlit and SymPy.")
