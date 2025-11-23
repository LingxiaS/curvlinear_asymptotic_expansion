import streamlit as st
import sympy as sp
from typing import List, Union

# --- Streamlit Page Configuration ---
st.set_page_config(page_title="Curvilinear Asymptotic Expansion Tool", layout="wide")

st.title("Curvilinear Asymptotic Expansion Tool")

# --- Sidebar / Input Area ---
st.sidebar.header("Calculation Parameters")

# 1. Select Method
method = st.sidebar.selectbox(
    "1. Select Calculation Method:",
    ("Laplacian", "CurvilinearDerivative")
)

# 2. Variable Symbol
variable_symbol_str = st.sidebar.text_input(
    "2. Variable Symbol (e.g., H):",
    value="H"
)

# 3 & 4. Expansion Terms
use_expansion = st.sidebar.checkbox("3. Use Expansion Variables", True)
expansion_terms_str = st.sidebar.text_area(
    "4. Expansion Terms (comma-separated, e.g., K(s), -rho*K(s)**2):",
    value="K(s), -rho*K(s)**2",
    disabled=not use_expansion
)

# 5. Expansion Order
order = st.sidebar.number_input(
    "5. Expansion Order (e.g., 2):",
    min_value=0,
    value=2
)

# 6. Components and Symbols
s_component_str = st.sidebar.text_input("sComponent Symbol (e.g., s):", value="s")
c_component_str = st.sidebar.text_input("cComponent Symbol (e.g., c):", value="c")


# --- Symbol Definition and Parsing ---

try:
    # Define necessary SymPy symbols
    s, c, rho = sp.symbols(f'{s_component_str} {c_component_str} rho')
    
    # Parse expansion terms
    if use_expansion and expansion_terms_str:
        expansion_terms_list: List[sp.Expr] = [sp.sympify(t.strip()) for t in expansion_terms_str.split(',')]
    else:
        expansion_terms_list: List[sp.Expr] = []

    variable_symbol = sp.Symbol(variable_symbol_str)

except Exception as e:
    st.error(f"Symbol/Expression Parsing Error: {e}")
    st.stop()


# --- Calculation Logic ---

@st.cache_data
def execute_calculation(
    method: str, 
    variable: sp.Symbol, 
    expansion_terms: List[sp.Expr], 
    order: int, 
    s_component: str, 
    c_component: str
) -> str:
    # Define Symbols needed by SymPy within the cached function
    s, c, rho = sp.symbols(f'{s_component} {c_component} rho')
    
    # Define a function H(s, c) for the variable
    H_func = sp.Function(variable.name)(s, c)
    
    if method == "Laplacian":
        # Final expanded result (Conceptual structure based on asymptotic expansion)
        final_result = sp.sympify(0)
        
        # Base Laplacian (H_ss + H_cc)
        base_laplacian = sp.Derivative(H_func, s, 2) + sp.Derivative(H_func, c, 2)
        final_result += base_laplacian
        
        # Incorporate the expansion terms
        for i in range(order + 1):
            if i < len(expansion_terms):
                 # Add the expansion term (this is where the actual complex logic resides)
                 final_result += expansion_terms[i] * (rho ** i)

        result_expr = final_result
        
    elif method == "CurvilinearDerivative":
        # Placeholder for the Curvilinear Derivative calculation
        result_expr = sp.Derivative(H_func, s) * c + rho
        
    else:
        raise ValueError("Invalid calculation method selected.")

    # Return the LaTeX representation of the final expression
    return sp.latex(result_expr)

# --- Execution Button and Result Display ---

if st.sidebar.button("Execute Calculation"):
    if not variable_symbol_str:
        st.warning("Please enter a Variable Symbol.")
    else:
        with st.spinner("Calculating..."):
            try:
                # Call core calculation function
                latex_result = execute_calculation(
                    method, 
                    variable_symbol, 
                    expansion_terms_list, 
                    order,
                    s_component_str,
                    c_component_str
                )

                st.success("Calculation Successful!")
                st.subheader("LaTeX Result:")
                
                # Display result using Streamlit's LaTeX function
                st.latex(latex_result)
                
            except Exception as e:
                st.error(f"Calculation Failed: {e}")
                st.exception(e)

st.markdown("---")
st.caption("Application powered by Streamlit and SymPy.")
