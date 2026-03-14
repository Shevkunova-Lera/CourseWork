import sys
import numpy as np
from scipy.optimize import linprog
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QGridLayout, QLabel, QLineEdit, QPushButton, QGroupBox, QMessageBox
)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont

# ============ CONFIGURATION (From coursework) ============
DEFAULT_DEMAND = [120, 140, 200, 150]       # Average demand
DEFAULT_PROD_MIN = [80, 120, 120, 100]      # Min. output (Normal)
DEFAULT_PROD_MAX = [100, 150, 210, 120]     # Max. output (Normal)
DEFAULT_COSTS = [8, 12, 10, 5, 15]          # [Norm, Over, Res, Storage, Penalty]
DEFAULT_LIMITS = [0.25, 0.30, 0.20]         # [Max Over%, Max Res%, Max Deficit Prob]

class TransportAssignmentSolver:
    """
    Implements the mathematical model for optimal production planning.
    Constructs the cost structure (similar to the Hungarian method) and solves it using linear programming.
    """
    def __init__(self):
        self.results = {}

    def solve(self, demand_mean, prod_min, prod_max, costs, limits):
        # 1. Parameter preparation
        c_norm, c_over, c_res, c_store, c_penalty = costs
        lim_over_pct, lim_res_pct, lim_def_prob = limits

        demand_mean = np.array(demand_mean, dtype=float)
        prod_min = np.array(prod_min, dtype=float)
        prod_max = np.array(prod_max, dtype=float)

        # Calculate average normal output
        avg_norm_prod = (prod_min + prod_max) / 2.0
        
        # Deterministic equivalent of demand
        alpha = 1.0 - lim_def_prob
        demand_quantile = -demand_mean * np.log(1.0 - alpha)

        # 2. Model building (Assignment matrix logic)
        num_periods = 4
        
        # Resource sources (Rows of the cost matrix)
        sources = [] 
        capacities = []
        
        for q in range(num_periods):
            # Normal
            sources.append((q, 'Norm'))
            capacities.append(avg_norm_prod[q])
            # Overtime
            sources.append((q, 'Over'))
            capacities.append(avg_norm_prod[q] * lim_over_pct)
            # Reserve
            sources.append((q, 'Res'))
            capacities.append(avg_norm_prod[q] * lim_res_pct)

        num_sources = len(sources)
        
        # Vectors for linear programming
        obj_c = []      # Objective function coefficients (costs)
        bounds = []     # Variable bounds
        linear_vars = [] # List of variables for building constraint matrices
        
        for i in range(num_sources):
            src_q, src_type = sources[i]
            
            # Determine base cost
            base_cost = 0
            if src_type == 'Norm': base_cost = c_norm
            elif src_type == 'Over': base_cost = c_over
            elif src_type == 'Res':  base_cost = c_res
            
            for j in range(num_periods):
                if src_q == j:
                    cost = base_cost
                elif src_q < j:
                    cost = base_cost + (j - src_q) * c_store
                else:
                    cost = 1e6 
                
                linear_vars.append({
                    'type': 'prod',
                    'src_idx': i,
                    'dst_idx': j,
                    'cost': cost,
                    'cap': capacities[i]
                })
                obj_c.append(cost)
                bounds.append((0, None))

        for j in range(num_periods):
            linear_vars.append({
                'type': 'deficit',
                'dst_idx': j,
                'cost': c_penalty,
                'cap': None
            })
            obj_c.append(c_penalty)
            bounds.append((0, None))

        # --- Forming constraints ---
        # 1. Supply constraints
        num_vars = len(linear_vars)
        A_ub = np.zeros((num_sources, num_vars))
        b_ub = np.zeros(num_sources)
        
        for r_idx, var in enumerate(linear_vars):
            if var['type'] == 'prod':
                A_ub[var['src_idx'], r_idx] = 1.0
                b_ub[var['src_idx']] = var['cap']
        
        # 2. Demand constraints
        A_eq = np.zeros((num_periods, num_vars))
        b_eq = demand_quantile 
        
        for r_idx, var in enumerate(linear_vars):
            j = var['dst_idx']
            A_eq[j, r_idx] = 1.0 
        
        # 3. Solving the problem (Simplex / Highs)
        res = linprog(obj_c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')

        # 4. Collecting results
        opt_norm = np.zeros(num_periods)
        opt_over = np.zeros(num_periods)
        opt_res  = np.zeros(num_periods)
        
        if res.success:
            x = res.x
            for r_idx, val in enumerate(x):
                var = linear_vars[r_idx]
                if var['type'] == 'prod' and val > 1e-5:
                    s_idx = var['src_idx']
                    q_prod, s_type = sources[s_idx]
                    
                    if s_type == 'Norm': opt_norm[q_prod] += val
                    elif s_type == 'Over': opt_over[q_prod] += val
                    elif s_type == 'Res':  opt_res[q_prod]  += val
            
            total_cost = res.fun
        else:
            total_cost = 0

        self.results = {
            "demand_quantile": demand_quantile,
            "avg_norm": avg_norm_prod,
            "opt_norm": opt_norm,
            "opt_over": opt_over,
            "opt_res": opt_res,
            "cost": total_cost,
        }
        return self.results

# ================== INTERFACE (GUI) ===================

class OptimizationWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.solver = TransportAssignmentSolver()
        self.fields = {}
        self.setup_window()
        self.build_interface()
        self.load_defaults()

    def setup_window(self):
        self.setWindowTitle("Optimal Planning (Transportation Model)")
        self.setMinimumWidth(750)
        self.setMinimumHeight(800) 
        self.setStyleSheet("""
            QMainWindow { background-color: #f0f2f5; }
            QGroupBox { font-weight: bold; border: 1px solid #bdc3c7; border-radius: 6px; margin-top: 10px; padding: 10px; background-color: white; }
            QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 5px; color: #2c3e50; }
            QLineEdit { border: 1px solid #dfe6e9; border-radius: 4px; padding: 4px; background-color: #fdfdfd; }
            QLineEdit:focus { border: 2px solid #3498db; }
            QPushButton { background-color: #3498db; color: white; border: none; padding: 8px 16px; border-radius: 4px; font-weight: bold; font-size: 14px; }
            QPushButton:hover { background-color: #2980b9; }
        """)

    def build_interface(self):
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)

        # === BLOCK 1: DATA ===
        grid_group = QGroupBox("Quarterly Data (Demand and Production)")
        grid = QGridLayout()
        headers = ["Quarter", "Avg Demand", "Min Output", "Max Output"]
        for c, h in enumerate(headers):
            grid.addWidget(QLabel(h), 0, c)

        self.fields["quarters"] = []
        for i in range(4):
            row_w = []
            grid.addWidget(QLabel(f"Q{i+1}"), i+1, 0)
            for j in range(3):
                le = QLineEdit()
                grid.addWidget(le, i+1, j+1)
                row_w.append(le)
            self.fields["quarters"].append(row_w)
        grid_group.setLayout(grid)
        layout.addWidget(grid_group)

        # === BLOCK 2: PARAMETERS ===
        params_layout = QHBoxLayout()
        
        cost_group = QGroupBox("Unit Costs (per unit)")
        cost_box = QVBoxLayout()
        self.fields["costs"] = []
        labels = ["Normal work", "Overtime", "Reserve", "Storage", "Penalty"]
        for l in labels:
            r = QHBoxLayout()
            r.addWidget(QLabel(l))
            le = QLineEdit()
            r.addWidget(le)
            cost_box.addLayout(r)
            self.fields["costs"].append(le)
        cost_group.setLayout(cost_box)
        params_layout.addWidget(cost_group)

        lim_group = QGroupBox("Constraints")
        lim_box = QVBoxLayout()
        self.fields["limits"] = []
        l_labels = ["Max % Overtime", "Max % Reserve", "Max Deficit Prob."]
        for l in l_labels:
            r = QHBoxLayout()
            r.addWidget(QLabel(l))
            le = QLineEdit()
            r.addWidget(le)
            lim_box.addLayout(r)
            self.fields["limits"].append(le)
        lim_group.setLayout(lim_box)
        params_layout.addWidget(lim_group)
        layout.addLayout(params_layout)

        # === BLOCK 3: BUTTON ===
        self.btn_calc = QPushButton("Calculate Optimal Plan")
        self.btn_calc.clicked.connect(self.calculate)
        layout.addWidget(self.btn_calc)

        # === BLOCK 4: RESULTS ===
        res_group = QGroupBox("Optimization Results")
        self.res_lbl = QLabel("Press 'Calculate'...")
        self.res_lbl.setFont(QFont("Courier New", 10))
        self.res_lbl.setAlignment(Qt.AlignCenter)
        v = QVBoxLayout()
        v.addWidget(self.res_lbl)
        res_group.setLayout(v)
        layout.addWidget(res_group)

    def load_defaults(self):
        for i in range(4):
            self.fields["quarters"][i][0].setText(str(DEFAULT_DEMAND[i]))
            self.fields["quarters"][i][1].setText(str(DEFAULT_PROD_MIN[i]))
            self.fields["quarters"][i][2].setText(str(DEFAULT_PROD_MAX[i]))
        for i, v in enumerate(DEFAULT_COSTS):
            self.fields["costs"][i].setText(str(v))
        for i, v in enumerate(DEFAULT_LIMITS):
            self.fields["limits"][i].setText(str(v))

    def get_data(self):
        try:
            d = [float(r[0].text()) for r in self.fields["quarters"]]
            p_min = [float(r[1].text()) for r in self.fields["quarters"]]
            p_max = [float(r[2].text()) for r in self.fields["quarters"]]
            c = [float(f.text()) for f in self.fields["costs"]]
            l = [float(f.text()) for f in self.fields["limits"]]
            return d, p_min, p_max, c, l
        except:
            QMessageBox.critical(self, "Error", "Please check the numbers!")
            return None

    def calculate(self):
        data = self.get_data()
        if not data: return
        
        try:
            res = self.solver.solve(*data)
            
            txt = f"MINIMUM COSTS: {res['cost']:,.2f}\n\n"
            txt += f"{'Quarter':<8} | {'Demand (qt)':<12} | {'Norm':<10} | {'Overtime':<10} | {'Reserve':<10}\n"
            txt += "-" * 65 + "\n"
            
            for i in range(4):
                txt += f" Q{i+1:<6} | {res['demand_quantile'][i]:<12.1f} | {res['opt_norm'][i]:<10.1f} | {res['opt_over'][i]:<10.1f} | {res['opt_res'][i]:<10.1f}\n"

            self.res_lbl.setText(txt)
            
        except Exception as e:
            QMessageBox.critical(self, "Calculation Error", str(e))

if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    win = OptimizationWindow()
    win.show()
    sys.exit(app.exec_())
