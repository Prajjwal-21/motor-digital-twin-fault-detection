function dxdt = dc_motor_ode(t, x, p, fault_type, severity, fault_onset)
%DC_MOTOR_ODE  Right-hand side of the DC motor digital twin.
%
%   State vector x = [i; w; Temp]
%       i    - armature current      [A]
%       w    - shaft speed           [rad/s]
%       Temp - winding temperature   [degC]
%
%   Equations:
%       L    di/dt = V - R*i - Ke*w                      (electrical)
%       J    dw/dt = Kt*i - B*w - T_L                    (mechanical)
%       C_th dT/dt = i^2*R + B*w^2 - (Temp - T_amb)/R_th (thermal)
%
%   Fault model (only active for t >= fault_onset):
%       fault_type = 0  Healthy             -> nothing changes
%       fault_type = 1  High Load           -> T_L is multiplied by severity
%       fault_type = 2  Increased Friction  -> B   is multiplied by severity

    i    = x(1);
    w    = x(2);
    Temp = x(3);

    % Start from the nominal (healthy) values, then apply the fault if active.
    T_L = p.T_L;
    B   = p.B;
    if t >= fault_onset
        if fault_type == 1
            T_L = p.T_L * severity;
        elseif fault_type == 2
            B = p.B * severity;
        end
    end

    di_dt = (p.V - p.R*i - p.Ke*w) / p.L;
    dw_dt = (p.Kt*i - B*w - T_L) / p.J;
    % Heat in: copper losses (i^2 R) + friction losses (B w^2).
    % Heat out: conduction to ambient through R_th.
    dT_dt = (i^2*p.R + B*w^2 - (Temp - p.T_amb)/p.R_th) / p.C_th;

    dxdt = [di_dt; dw_dt; dT_dt];
end
