function p = motor_params()
%MOTOR_PARAMS  Nominal parameters of a small 24 V brushed DC motor.
%
%   p = motor_params() returns a struct with the electrical, mechanical and
%   thermal parameters used by the digital twin (see dc_motor_ode.m).
%
%   At 24 V the motor settles at roughly 3400 RPM and draws about 1.3 A.

    % ---------------- Electrical ----------------
    p.V     = 24;        % supply voltage                       [V]
    p.R     = 2.0;       % armature resistance                  [Ohm]
    p.L     = 5e-3;      % armature inductance                  [H]
    p.Ke    = 0.06;      % back-EMF constant                    [V/(rad/s)]

    % ---------------- Mechanical ----------------
    p.Kt    = 0.06;      % torque constant (= Ke in SI units)   [N*m/A]
    p.J     = 2e-4;      % rotor + load inertia                 [kg*m^2]
    p.B     = 1e-4;      % viscous friction coefficient         [N*m*s/rad]
    p.T_L   = 0.04;      % constant load torque                 [N*m]

    % ---------------- Thermal -------------------
    % A real motor of this size has a thermal time constant of several
    % minutes (C_th ~ 50 J/K, R_th ~ 10 K/W -> tau ~ 500 s). That is far too
    % slow to see anything in a 10 s simulation, so the constants are SCALED
    % DOWN on purpose: tau = C_th * R_th = 2 * 3 = 6 s. The temperature then
    % rises by tens of degrees within the 10 s window, which makes the
    % heating effect of each fault visible to the classifier.
    p.C_th  = 2.0;       % thermal capacitance (scaled)         [J/K]
    p.R_th  = 3.0;       % thermal resistance to ambient        [K/W]
    p.T_amb = 25;        % ambient temperature                  [degC]
end
