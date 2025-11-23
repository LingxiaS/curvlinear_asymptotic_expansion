import streamlit as st
import sympy as sp
from typing import List, Union, Tuple

# --- SymPy Core Calculation Functions (STRICTLY AS PROVIDED BY USER) ---

def curvilinear_gradient(phi, n_vec, s_vec, epsilon, rho, s, K, order):
    """Calculates the curvilinear gradient (nabla phi). Order in series should be user_order + 1."""
    # Ensure series order is user_order + 1 for accuracy up to user_order
    series_order = order + 1 
    
    grad_rho = sp.expand(n_vec * (1/epsilon) * sp.diff(phi, rho))
    # Using the user-specified series order for the term: (1/(1 + epsilon*rho*K))
    grad_s   = sp.expand(s_vec * (1/(1 + epsilon*rho*K)).series(epsilon, 0, series_order) * sp.diff(phi, s))
    
    return sp.expand(grad_rho + grad_s)


def curvilinear_divergence(n_component, s_component, epsilon, rho, s, K, order):
    """Calculates the curvilinear divergence (div(V)). Order in series should be user_order + 1."""
    # Ensure series order is user_order + 1 for accuracy up to user_order
    series_order = order + 1
    
    n_dot_vec = n_component
    s_dot_vec = s_component
    
    term1 = sp.expand((1/epsilon) * sp.diff(n_dot_vec, rho))
    # Using the user-specified series order for the term: (1/(1 + epsilon*rho*K))
    term2 = sp.expand( (1/(1 + epsilon*rho*K)).series(epsilon, 0, series_order) * sp.diff(s_dot_vec, s))
    term3 = sp.expand((1/(1 + epsilon*rho*K)).series(epsilon, 0, series_order) * K * n_dot_vec)
    
    div = sp.collect(term1 + term2 + term3, epsilon)
    return div

def curvilinear_laplacian(phi, n_vec, s_vec, epsilon, rho, s, K, order):
    """Calculates the curvilinear Laplacian (nabla^2 phi)."""
    # The order passed to sub-functions is the user's intended final order.
    grad_phi = curvilinear_gradient(phi, n_vec, s_vec, epsilon, rho, s, K, order)
    
    n_comp_grad = grad_phi.subs({s_vec: 0, n_vec: 1}) 
    s_comp_grad = grad_phi.subs({s_vec: 1, n_vec: 0})
    
    lap_phi = curvilinear_divergence(n_comp_grad, s_comp_grad, epsilon, rho, s, K, order)

    return sp.collect(lap_phi, epsilon)


# --- Streamlit Page Configuration ---
st.set_page_config(page_title="Curvilinear Asymptotic Expansion Tool", layout="wide")

st.title("Curvilinear Asymptotic Expansion Tool")
st.markdown("---")


# --- Main Instruction and Formula (FIX: All UI/Formula Issues) ---

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
# FIX: Corrected mathematical formula display
st.latex(r'''
\nabla \cdot \mathbf{V} = \frac{1}{\epsilon} \frac{\partial V_n}{\partial \rho} + \frac{1}{1 + \epsilon \rho K} \left( \frac{\partial V_s}{\partial s} + K V_n \right)
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

# 2. Variable Type (FIX: Cleaned dropdown options)
variable_type = st.sidebar.selectbox(
    "2. Variable Type:",
    ("Scalar", "Vector")
)

# 3. Apply Which Expansion (FIX: Cleaned dropdown options)
if variable_type == "Scalar":
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
if variable_type == "Vector":
    st.sidebar.subheader("Vector Components")
    n_component_str = st.sidebar.text_input(
        f"Normal Component ($V_n$):",
        value="0"
    )
    s_component_str = st.sidebar.text_input(
        f"Tangential Component ($V_s$):",
        value="0"
    )

# 5. Expansion Order
order = st.sidebar.number_input(
    "4. Expansion Order (n, result up to $\\epsilon^n$):",
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
        pass 


# --- Core Execution Function ---

@st.cache_data
def execute_calculation(
    op_type: str, 
    variable_name: str,
    v_type: str,
    n_comp_str: str,
    s_comp_str: str,
    _expansion_terms: List[sp.Expr], 
    order: int
) -> Tuple[str, str, int]:
    
    # 1. Define base symbols
    rho, s, epsilon = sp.symbols('rho s epsilon')
    K = sp.Function('K')(s)
    
    # Use bold LaTeX symbols for basis vectors
    n_vec, s_vec = sp.symbols(r'\mathbf{n} \mathbf{s}') 

    # 2. Define the Variable (H or V)
    if v_type == "Scalar":
        if _expansion_terms:
            # Construct expression and add Order term for correct SymPy series handling
            phi = sp.sympify(0)
            for i, term in enumerate(_expansion_terms):
                phi += term * (epsilon ** i)
            # Add O(eps^(order+1)) to ensure the result is correctly truncated at order 'order'
            if len(_expansion_terms) <= order + 1:
                 phi += sp.Order(epsilon**(order + 1))
        else:
            phi = sp.Function(variable_name)(rho, s)
            
        target_expr = phi
        
    elif v_type == "Vector":
        Vn = sp.sympify(n_comp_str)
        Vs = sp.sympify(s_comp_str)
        target_expr = Vn * n_vec + Vs * s_vec
        
    else:
        raise ValueError("Invalid variable type.")

    # 3. Perform the calculation based on the selected operator
    if op_type == "Gradient ($\nabla$)" and v_type == "Scalar":
        result_expr = curvilinear_gradient(target_expr, n_vec, s_vec, epsilon, rho, s, K, order)
        op_symbol = r"\nabla " + variable_name
        
    elif op_type == "Laplacian ($\nabla^2$)" and v_type == "Scalar":
        result_expr = curvilinear_laplacian(target_expr, n_vec, s_vec, epsilon, rho, s, K, order)
        op_symbol = r"\nabla^2 " + variable_name
        
    elif op_type == "Divergence ($\nabla \cdot$)" and v_type == "Vector":
        result_expr = curvilinear_divergence(Vn, Vs, epsilon, rho, s, K, order) 
        op_symbol = r"\nabla \cdot \mathbf{" + variable_name + "}"
        
    else:
        raise ValueError("Invalid operation/type combination.")

    # 4. Final Truncation: Truncate at O(eps^(order+1))
    # This removes all terms >= epsilon^(order+1)
    final_result = sp.collect(result_expr.subs(sp.Order(epsilon**(order + 1)), 0), epsilon)

    return op_symbol, sp.latex(final_result), order


# --- Execution and Display ---

st.subheader("Result")

if st.sidebar.button("Execute Calculation"):
    if variable_type == "Vector" and not (n_component_str or s_component_str):
         st.warning("Please define at least one vector component ($V_n$ or $V_s$).")
    elif not variable_name:
        st.warning("Please enter a Variable Name.")
    else:
        with st.spinner(f"Calculating {operation} of {variable_name}..."):
            try:
                op_symbol, latex_result, final_order = execute_calculation(
                    operation, 
                    variable_name,
                    variable_type,
                    n_component_str,
                    s_component_str,
                    expansion_terms, 
                    order
                )

                st.success(f"Calculation Successful for ${op_symbol}$")
                
                # Append O(eps^(order+1)) to the result display
                st.subheader(f"Expanded Form of ${op_symbol}$ (Up to $\\epsilon^{final_order}$):")
                st.latex(latex_result + r" + \mathcal{O}\left(\epsilon^{%s}\right)" % (final_order + 1))
                
            except Exception as e:
                st.error(f"Calculation Failed: Check your input expressions.")
                st.exception(e)

st.markdown("---")


# --- Example Demo Usage ---
st.header("Example: Laplacian of Expanded H")
st.markdown("""
This demo replicates a common use case where the variable $H$ is expanded up to $O(\epsilon^1)$ ($H = H_0 + \epsilon H_1$).

**Inputs used in this example:**
* **Variable Name:** H
* **Variable Type:** Scalar
* **Operator:** Laplacian ($\nabla^2$)
* **Expansion Order:** 2
* **Expansion Terms:** `K(s)`, `-rho * K(s)**2` (i.e., $H_0 = K(s)$, $H_1 = -\\rho K(s)^2$)
""")

# Example variables defined in the same way as the main app
ex_rho, ex_s, ex_epsilon = sp.symbols('rho s epsilon')
ex_K = sp.Function('K')(ex_s)
ex_n_vec, ex_s_vec = sp.symbols(r'\mathbf{n} \mathbf{s}') 

# Expansion: H0 + eps*H1
ex_H_0 = ex_K
ex_H_1 = -ex_rho * ex_K**2
# Ensure expansion is done to O(eps^(2+1)) for accurate Laplacian calculation (order=2)
ex_H_expanded = ex_H_0 + ex_H_1 * ex_epsilon + sp.Order(ex_epsilon**(2+1)) 

# Calculate the Laplacian of the expanded H (using order=2)
ex_lap_H = curvilinear_laplacian(ex_H_expanded, ex_n_vec, ex_s_vec, ex_epsilon, ex_rho, ex_s, ex_K, 2)
# Final Truncation at O(eps^(2+1))
ex_lap_H_final = sp.collect(ex_lap_H.subs(sp.Order(ex_epsilon**3), 0), ex_epsilon)

st.subheader("Result for $\\nabla^2 H$:")
# Append O(eps^3) to the example result display
st.latex(sp.latex(ex_lap_H_final) + r" + \mathcal{O}\left(\epsilon^3\right)")

st.caption("Application powered by Streamlit and SymPy.")
