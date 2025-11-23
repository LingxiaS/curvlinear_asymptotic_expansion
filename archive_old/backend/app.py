import sympy as sp
from flask import Flask, request, jsonify
from flask_cors import CORS 

app = Flask(__name__)
# Enable Cross-Origin Resource Sharing (CORS) for the frontend
#CORS(app)
# Explicitly configure CORS: allow all origins ("*") to access all routes (resources={r"/*"})
CORS(app, resources={r"/*": {"origins": "*"}}) 
app.config['CORS_HEADERS'] = 'Content-Type'

# --- Core Calculus Functions ---

# Symbol definition (Global)
# K is a function of s, phi is a function of rho, s
rho, s, epsilon = sp.symbols('rho s epsilon', real=True)
K = sp.Function('K')(s)
n_vec = sp.symbols('n_vec')
s_vec = sp.symbols('s_vec')


def curvilinear_gradient(phi, n_vec, s_vec, epsilon, rho, s, K, order):
    """Calculates the gradient in curvilinear coordinates, and performs series expansion for 1/(1+epsilon*rho*K)."""
    grad_rho = sp.expand(n_vec * (1/epsilon) * sp.diff(phi, rho))
    # Expand (1/(1 + epsilon*rho*K)) up to the specified order
    series_term = (1/(1 + epsilon*rho*K)).series(epsilon, 0, order).removeO()
    grad_s = sp.expand(s_vec * series_term * sp.diff(phi, s))
    return sp.expand(grad_rho + grad_s)


def curvilinear_divergence(n_component, s_component, epsilon, rho, s, K):
    """Calculates the divergence in curvilinear coordinates."""
    n_dot_vec = n_component
    s_dot_vec = s_component
    
    # Term 1: (1/epsilon) * dV_n/drho
    term1 = sp.expand((1/epsilon) * sp.diff(n_dot_vec, rho))
    
    # Term 2: (1/(1 + epsilon*rho*K)) * dV_s/ds
    term2_geom = 1 / (1 + epsilon*rho*K)
    term2 = sp.expand(term2_geom * sp.diff(s_dot_vec, s))
    
    # Term 3: K * V_n / (1 + epsilon*rho*K)
    term3 = sp.expand(K * n_dot_vec / (1 + epsilon*rho*K))
    
    div = sp.collect(term1 + term2 + term3, epsilon)
    return div


def curvilinear_laplacian(phi, n_vec, s_vec, epsilon, rho, s, K, order):
    """Calculates the Laplacian operator."""
    grad_phi = curvilinear_gradient(phi, n_vec, s_vec, epsilon, rho, s, K, order)
    
    # Extract n_vec and s_vec components using substitution
    n_comp = grad_phi.subs({s_vec: 0, n_vec: 1})
    s_comp = grad_phi.subs({s_vec: 1, n_vec: 0})
    
    lap_phi = curvilinear_divergence(n_comp, s_comp, epsilon, rho, s, K)

    return sp.collect(lap_phi, epsilon)


# --- API Route ---

@app.route('/calculate', methods=['POST'])
def calculate():
    data = request.get_json()
    
    # 1. Extract input parameters
    op_type = data.get('operation')
    var_symbol_str = data.get('variableSymbol')
    expansion_order = int(data.get('order', 2)) 
    
    # Flag for variable expansion
    do_variable_expansion = data.get('doVariableExpansion') == 'true'
    
    # 2. Define variable symbol (using user input)
    var_symbol = sp.Function(var_symbol_str)(rho, s)
    
    # 3. Handle Variable Expansion (e.g., phi = phi0 + eps*phi1 + ...)
    variable_expr = var_symbol
    if do_variable_expansion:
        try:
            expansion_terms_str = data.get('expansionTerms', '')
            
            # Allow user to input string like "V0, V1"
            term_strings = [t.strip() for t in expansion_terms_str.split(',') if t.strip()]
            
            if not term_strings:
                raise ValueError("Please provide at least a zero-order term (e.g., V0) for expansion.")
            
            # Construct the series
            expansion_series = sp.sympify(term_strings[0]) 
            
            max_epsilon_power = len(term_strings) 
            
            for i in range(1, len(term_strings)):
                expansion_series += sp.sympify(term_strings[i]) * (epsilon**i)
            
            # Substitute the original variable and truncate
            variable_expr = expansion_series + sp.Order(epsilon**max_epsilon_power, epsilon)
            
        except Exception as e:
            return jsonify({'error': f"Error parsing variable expansion expression: {e}"}), 400

    
    # 4. Perform calculation
    result_expr = None
    try:
        if op_type == 'Gradient':
            # Gradient is for scalars
            result_expr = curvilinear_gradient(
                variable_expr, n_vec, s_vec, epsilon, rho, s, K, expansion_order
            )
            
        elif op_type == 'Laplacian':
            # Laplacian is for scalars
            result_expr = curvilinear_laplacian(
                variable_expr, n_vec, s_vec, epsilon, rho, s, K, expansion_order
            )
            
        elif op_type == 'Divergence':
            # Divergence is for vectors
            n_comp_str = data.get('nComponent', '0')
            s_comp_str = data.get('sComponent', '0')
            
            # User input components might already include epsilon terms
            n_comp_expr = sp.sympify(n_comp_str)
            s_comp_expr = sp.sympify(s_comp_str)
            
            result_expr = curvilinear_divergence(
                n_comp_expr, s_comp_expr, epsilon, rho, s, K
            )
            
        else:
            return jsonify({'error': 'Unsupported calculation operation type'}), 400

        # 5. Format output (using SymPy's LaTeX format)
        # Convert to LaTeX string
        latex_output = sp.latex(result_expr, full_paren=True)
        
        # 6. Return result
        return jsonify({'result_latex': latex_output})

    except Exception as e:
        # Catch any errors during calculation
        return jsonify({'error': f"An error occurred during calculation: {e}"}), 500


if __name__ == '__main__':
    # Default run on http://127.0.0.1:5000/
    app.run(debug=True)



@app.route('/')
def health_check():
    # Return a simple JSON response to confirm the service is running
    return jsonify({'status': 'API Running', 'version': '1.0'})

