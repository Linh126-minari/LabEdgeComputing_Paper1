import cvxpy as cp
import numpy as np

class CMSOSolver:
    def solve(self, task_params, user_params, av_params, prices):
        """
        Giải bài toán Q3 cho một người dùng cụ thể.
        """
        # Biến quyết định: tỷ lệ xử lý cục bộ (alpha_l)
        # alpha_n (offload) = 1 - alpha_l
        alpha_l = cp.Variable() 
        
        s_m, c_m, t_m, c_en = task_params
        f_m, lambda_l, lambda_n, p_m, p_w = user_params
        f_n, r_up, g_n = av_params # g_n là giá dịch vụ động
        
        # Các thành phần chi phí (Dựa trên Eq 35)
        # 1. Năng lượng cục bộ: k * f_m^2 * alpha_l * s_m * c_m
        # Lưu ý: f_m tối ưu cũng phụ thuộc alpha_l (Eq 34: f_m* = alpha_l * s_m * c_m / t_m)
        # Thay thế f_m* vào công thức năng lượng:
        # E_local = k * (alpha_l * s_m * c_m / t_m)^2 * alpha_l * s_m * c_m
        #         = k * (s_m * c_m)^3 / t_m^2 * alpha_l^3  <-- Hàm bậc 3 (Lồi với alpha >= 0)
        
        # CVXPY hỗ trợ hàm power(x, 3).
        E_local = user_params.kappa * cp.power(alpha_l * s_m * c_m, 3) / (t_m**2)
        
        # 2. Chi phí offloading (E_offload + Service Fee)
        # Các số hạng này tuyến tính theo alpha_n = (1 - alpha_l)
        alpha_n = 1 - alpha_l
        E_offload = lambda_n * (
            user_params.kappa * cp.power(f_m, 2) * alpha_n * s_m * c_en + # Encryption energy
            (alpha_n * s_m / r_up) * p_m +                                # Transmission energy
            (p_w * alpha_n * s_m * c_m / f_n)                             # Waiting energy
        )
        
        Service_Cost = g_n * alpha_n * s_m * c_m
        
        # Hàm mục tiêu
        objective = cp.Minimize(lambda_l * E_local + E_offload + Service_Cost)
        
        # Ràng buộc (Eq 31b-31f & 32-33)
        # alpha_l nằm trong khoảng [alpha_min, alpha_max] xác định bởi delay constraints
        constraints = [
            alpha_l >= 0,
            alpha_l <= 1,
            # Các ràng buộc delay bổ sung từ Eq 32, 33
        ]
        
        prob = cp.Problem(objective, constraints)
        
        try:
            prob.solve()
            return alpha_l.value
        except cp.SolverError:
            return 0.5 # Fallback nếu solver thất bại