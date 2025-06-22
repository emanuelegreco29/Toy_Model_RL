import numpy as np
from scipy.interpolate import CubicSpline, UnivariateSpline
from scipy.optimize import minimize_scalar
from collections import deque

class BSplinePredictor:
    """
    Advanced predictor for B-spline trajectories with multiple prediction methods
    """
    
    def __init__(self, method='spline_fit', window_size=10):
        """
        Args:
            method: 'spline_fit', 'polynomial', 'velocity_based', 'curvature_aware'
            window_size: Number of recent points to use for prediction
        """
        self.method = method
        self.window_size = window_size
        self.history = deque(maxlen=window_size)
        
    def update_history(self, point):
        """Add new point to history"""
        self.history.append(point.copy())
    
    def predict_spline_fit(self, horizon=1):
        """
        Fit cubic spline to recent history and extrapolate
        Most accurate for smooth B-spline trajectories
        """
        if len(self.history) < 4:  # Need at least 4 points for cubic spline
            return self._fallback_prediction(horizon)
        
        history = np.array(self.history)
        n_points = len(history)
        
        # Create time parameter (normalized)
        t = np.linspace(0, 1, n_points)
        
        # Fit spline for each dimension
        splines = []
        for dim in range(3):
            try:
                # Use smoothing spline to handle noise
                spline = UnivariateSpline(t, history[:, dim], s=0.1)
                splines.append(spline)
            except:
                # Fallback to linear if spline fails
                return self._fallback_prediction(horizon)
        
        # Extrapolate
        dt = 1.0 / (n_points - 1)  # Time step in normalized coordinates
        t_future = 1.0 + horizon * dt
        
        prediction = np.array([spline(t_future) for spline in splines])
        return prediction.astype(np.float32)
    
    def predict_polynomial(self, horizon=1, degree=3):
        """
        Fit polynomial to trajectory and extrapolate
        Good for smooth trajectories with consistent curvature
        """
        if len(self.history) < degree + 1:
            return self._fallback_prediction(horizon)
        
        history = np.array(self.history)
        n_points = len(history)
        
        # Time parameter
        t = np.arange(n_points, dtype=np.float32)
        
        prediction = np.zeros(3, dtype=np.float32)
        
        for dim in range(3):
            try:
                # Fit polynomial
                coeffs = np.polyfit(t, history[:, dim], degree)
                poly = np.poly1d(coeffs)
                
                # Predict
                t_future = n_points - 1 + horizon
                prediction[dim] = poly(t_future)
            except:
                # Fallback to linear
                if n_points >= 2:
                    velocity = (history[-1, dim] - history[-2, dim])
                    prediction[dim] = history[-1, dim] + horizon * velocity
                else:
                    prediction[dim] = history[-1, dim]
        
        return prediction
    
    def predict_velocity_based(self, horizon=1, smoothing=True):
        """
        Use velocity and acceleration estimation
        Good for trajectories with varying speed
        """
        if len(self.history) < 3:
            return self._fallback_prediction(horizon)
        
        history = np.array(self.history)
        n_points = len(history)
        
        if smoothing and n_points >= 5:
            # Smooth velocity estimation using central differences
            velocities = []
            for i in range(2, n_points-2):
                # 5-point stencil for smooth derivative
                vel = (-history[i+2] + 8*history[i+1] - 8*history[i-1] + history[i-2]) / 12.0
                velocities.append(vel)
            
            if len(velocities) >= 2:
                # Smooth acceleration
                accel = velocities[-1] - velocities[-2]
                current_vel = velocities[-1]
            else:
                current_vel = history[-1] - history[-2]
                accel = np.zeros(3)
        else:
            # Simple finite differences
            current_vel = history[-1] - history[-2]
            if n_points >= 3:
                prev_vel = history[-2] - history[-3]
                accel = current_vel - prev_vel
            else:
                accel = np.zeros(3)
        
        # Predict using kinematic equation
        current_pos = history[-1]
        prediction = current_pos + horizon * current_vel + 0.5 * horizon**2 * accel
        
        return prediction.astype(np.float32)
    
    def predict_curvature_aware(self, horizon=1):
        """
        Advanced prediction considering trajectory curvature
        Best for complex B-spline paths with varying curvature
        """
        if len(self.history) < 6:
            return self._fallback_prediction(horizon)
        
        history = np.array(self.history)
        n_points = len(history)
        
        # Estimate local curvature and torsion
        # Use finite differences for derivatives
        
        # First derivatives (velocity)
        v1 = np.gradient(history[:, 0])
        v2 = np.gradient(history[:, 1]) 
        v3 = np.gradient(history[:, 2])
        velocity = np.column_stack([v1, v2, v3])
        
        # Second derivatives (acceleration)
        a1 = np.gradient(v1)
        a2 = np.gradient(v2)
        a3 = np.gradient(v3)
        acceleration = np.column_stack([a1, a2, a3])
        
        # Current velocity and acceleration
        curr_vel = velocity[-1]
        curr_acc = acceleration[-1]
        
        # Estimate curvature (κ = |v × a| / |v|³)
        cross_product = np.cross(curr_vel, curr_acc)
        speed = np.linalg.norm(curr_vel)
        
        if speed > 1e-6:
            curvature = np.linalg.norm(cross_product) / (speed**3)
        else:
            curvature = 0.0
        
        # Predict using Frenet-Serret formulas approximation
        current_pos = history[-1]
        
        if speed > 1e-6:
            # Unit tangent vector
            tangent = curr_vel / speed
            
            # Approximate normal vector (points toward center of curvature)
            if np.linalg.norm(curr_acc) > 1e-6:
                normal_component = curr_acc - np.dot(curr_acc, tangent) * tangent
                if np.linalg.norm(normal_component) > 1e-6:
                    normal = normal_component / np.linalg.norm(normal_component)
                else:
                    normal = np.zeros(3)
            else:
                normal = np.zeros(3)
            
            # Predict position considering curvature
            # Simple arc approximation
            if curvature > 1e-6:
                radius = 1.0 / curvature
                arc_length = speed * horizon
                angle = arc_length / radius
                
                # Rotation in the osculating plane
                cos_angle = np.cos(angle)
                sin_angle = np.sin(angle)
                
                # Approximate curved motion
                linear_motion = horizon * curr_vel
                curvature_correction = radius * sin_angle * normal - radius * (1 - cos_angle) * tangent
                
                prediction = current_pos + linear_motion + curvature_correction
            else:
                # Fall back to linear + acceleration
                prediction = current_pos + horizon * curr_vel + 0.5 * horizon**2 * curr_acc
        else:
            prediction = current_pos
        
        return prediction.astype(np.float32)
    
    def predict_adaptive(self, horizon=1):
        """
        Adaptive prediction that chooses the best method based on trajectory characteristics
        """
        if len(self.history) < 4:
            return self._fallback_prediction(horizon)
        
        history = np.array(self.history)
        
        # Analyze trajectory characteristics
        velocities = np.diff(history, axis=0)
        speeds = np.linalg.norm(velocities, axis=1)
        
        # Check if trajectory is relatively straight
        if len(speeds) >= 3:
            speed_variation = np.std(speeds) / (np.mean(speeds) + 1e-6)
            direction_changes = []
            
            for i in range(1, len(velocities)):
                if np.linalg.norm(velocities[i]) > 1e-6 and np.linalg.norm(velocities[i-1]) > 1e-6:
                    cos_angle = np.dot(velocities[i], velocities[i-1]) / (
                        np.linalg.norm(velocities[i]) * np.linalg.norm(velocities[i-1])
                    )
                    cos_angle = np.clip(cos_angle, -1, 1)
                    angle_change = np.arccos(cos_angle)
                    direction_changes.append(angle_change)
            
            if direction_changes:
                avg_direction_change = np.mean(direction_changes)
            else:
                avg_direction_change = 0.0
            
            # Choose method based on characteristics
            if speed_variation < 0.1 and avg_direction_change < 0.2:
                # Straight trajectory with constant speed
                return self.predict_velocity_based(horizon, smoothing=False)
            elif avg_direction_change > 0.5:
                # High curvature trajectory
                return self.predict_curvature_aware(horizon)
            else:
                # Moderate curvature - use spline fitting
                return self.predict_spline_fit(horizon)
        
        return self.predict_spline_fit(horizon)
    
    def _fallback_prediction(self, horizon=1):
        """Simple linear extrapolation fallback"""
        if len(self.history) < 2:
            return self.history[-1].copy() if self.history else np.zeros(3, dtype=np.float32)
        
        history = np.array(self.history)
        velocity = history[-1] - history[-2]
        prediction = history[-1] + horizon * velocity
        return prediction.astype(np.float32)
    
    def predict(self, horizon=1):
        """Main prediction function using selected method"""
        if self.method == 'spline_fit':
            return self.predict_spline_fit(horizon)
        elif self.method == 'polynomial':
            return self.predict_polynomial(horizon)
        elif self.method == 'velocity_based':
            return self.predict_velocity_based(horizon)
        elif self.method == 'curvature_aware':
            return self.predict_curvature_aware(horizon)
        elif self.method == 'adaptive':
            return self.predict_adaptive(horizon)
        else:
            return self._fallback_prediction(horizon)