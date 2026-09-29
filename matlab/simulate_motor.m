function tbl = simulate_motor(p, fault_type, severity, fault_onset, noise_level)
%SIMULATE_MOTOR  Run the DC motor digital twin for 10 s and return a table.
%
%   tbl = simulate_motor(p, fault_type, severity, fault_onset)
%   tbl = simulate_motor(p, fault_type, severity, fault_onset, noise_level)
%
%   Inputs
%       p           - parameter struct from motor_params()
%       fault_type  - 0 Healthy, 1 High Load, 2 Increased Friction
%       severity    - fault multiplier (e.g. 1.3 = +30%)
%       fault_onset - time [s] at which the fault starts
%       noise_level - relative std of Gaussian sensor noise (default 0.01,
%                     i.e. 1% of each signal's typical size; 0 = no noise)
%
%   Output: table with 100 rows and columns
%       time [s], current [A], speed_rpm [RPM], torque [N*m], temperature [degC]

    if nargin < 5
        noise_level = 0.01;
    end

    t_end   = 10;                        % simulation length [s]
    n_steps = 100;                       % samples fed to the neural networks
    x0      = [0; 0; p.T_amb];           % motor starts at rest, at ambient temp

    % ode45 (adaptive Runge-Kutta 4/5) is accurate and fast enough here: the
    % electrical time constant L/R = 2.5 ms is not stiff enough to need ode15s.
    % MaxStep keeps the solver from stepping over the fault switch-on.
    opts = odeset('RelTol', 1e-5, 'AbsTol', 1e-7, 'MaxStep', 0.01);
    odefun = @(t, x) dc_motor_ode(t, x, p, fault_type, severity, fault_onset);
    [t_raw, x_raw] = ode45(odefun, [0 t_end], x0, opts);

    % ode45 returns an irregular time grid -> resample to exactly 100 points.
    time = linspace(0, t_end, n_steps)';
    x    = interp1(t_raw, x_raw, time);

    current     = x(:, 1);
    speed_rpm   = x(:, 2) * 60 / (2*pi);  % rad/s -> RPM
    torque      = p.Kt * current;         % electromagnetic torque
    temperature = x(:, 3);

    % Sensor noise: each signal gets Gaussian noise whose std is noise_level
    % times that signal's mean absolute value (so 1% noise is 1% for every
    % channel regardless of units).
    add_noise = @(s) s + noise_level * mean(abs(s)) * randn(size(s));
    current     = add_noise(current);
    speed_rpm   = add_noise(speed_rpm);
    torque      = add_noise(torque);
    temperature = add_noise(temperature);

    tbl = table(time, current, speed_rpm, torque, temperature);
end
