function generate_dataset()
%GENERATE_DATASET  Simulate 900 motor runs (300 per class) and save to CSV.
%
%   Classes:  0 Healthy | 1 High Load (T_L up) | 2 Increased Friction (B up)
%
%   To make the task realistic (and not trivially 100% solvable), every run
%   gets its own operating point and its own fault timing/strength:
%     * fault onset   : uniform 2 - 6 s
%     * fault severity: uniform 1.3x (mild) - 3x (strong)
%     * supply voltage: uniform 20 - 28 V
%     * nominal load  : +-30% around motor_params().T_L
%     * nominal friction: +-30% around motor_params().B
%     * ambient temp  : uniform 20 - 35 degC
%
%   Outputs
%     data/motor_dataset.csv  run_id, time, current, speed, torque, temperature, label
%     data/run_metadata.csv   one row per run with the randomized settings
%                             (used for analysis / plotting only, NOT for training)

    rng(42);                                   % reproducible dataset

    % ---- Dataset settings (all "difficulty knobs" live here) ----
    cfg.runs_per_class = 300;
    cfg.onset_range    = [2 6];                % [s]
    cfg.severity_range = [1.3 3.0];            % fault multiplier
    cfg.voltage_range  = [20 28];              % [V]
    cfg.load_spread    = 0.30;                 % +-30% nominal load
    cfg.friction_spread= 0.30;                 % +-30% nominal friction
    cfg.ambient_range  = [20 35];              % [degC]
    cfg.noise_level    = 0.01;                 % 1% sensor noise

    class_names = ["Healthy", "High Load", "Increased Friction"];
    n_classes   = numel(class_names);
    n_runs      = n_classes * cfg.runs_per_class;

    root     = fileparts(fileparts(mfilename('fullpath')));   % project root
    data_dir = fullfile(root, 'data');
    if ~exist(data_dir, 'dir'); mkdir(data_dir); end

    base      = motor_params();
    run_tbls  = cell(n_runs, 1);
    meta      = table('Size', [n_runs 7], ...
        'VariableTypes', repmat("double", 1, 7), ...
        'VariableNames', ["run_id" "label" "fault_onset" "severity" ...
                          "voltage" "load_torque" "friction"]);
    meta = addvars(meta, zeros(n_runs, 1), 'NewVariableNames', 'ambient');

    uniform = @(r) r(1) + (r(2) - r(1)) * rand();   % helper: U(r(1), r(2))

    fprintf('Generating %d runs (%d per class)...\n', n_runs, cfg.runs_per_class);
    tic;
    run_id = 0;
    for label = 0:n_classes-1
        for k = 1:cfg.runs_per_class
            run_id = run_id + 1;

            % Randomize the operating point of this particular motor/run.
            p       = base;
            p.V     = uniform(cfg.voltage_range);
            p.T_L   = base.T_L * uniform([1 - cfg.load_spread, 1 + cfg.load_spread]);
            p.B     = base.B   * uniform([1 - cfg.friction_spread, 1 + cfg.friction_spread]);
            p.T_amb = uniform(cfg.ambient_range);

            onset = uniform(cfg.onset_range);
            if label == 0
                severity = 1.0;                % healthy: no fault
            else
                severity = uniform(cfg.severity_range);
            end

            tbl = simulate_motor(p, label, severity, onset, cfg.noise_level);

            % Rename speed_rpm -> speed for the CSV and add run_id / label.
            tbl = renamevars(tbl, 'speed_rpm', 'speed');
            tbl.run_id = repmat(run_id, height(tbl), 1);
            tbl.label  = repmat(label,  height(tbl), 1);
            run_tbls{run_id} = tbl(:, ["run_id" "time" "current" "speed" ...
                                       "torque" "temperature" "label"]);

            meta(run_id, :) = {run_id, label, onset, severity, ...
                               p.V, p.T_L, p.B, p.T_amb};

            if mod(run_id, 50) == 0
                fprintf('  %4d / %d runs done (%.1f s)\n', run_id, n_runs, toc);
            end
        end
    end

    dataset = vertcat(run_tbls{:});
    csv_path = fullfile(data_dir, 'motor_dataset.csv');
    writetable(dataset, csv_path);
    writetable(meta, fullfile(data_dir, 'run_metadata.csv'));

    fprintf('\nSaved %s (%d rows, %d runs) in %.1f s\n', ...
            csv_path, height(dataset), n_runs, toc);
    fprintf('Class counts (runs):\n');
    for label = 0:n_classes-1
        fprintf('  %d  %-20s %d\n', label, class_names(label+1), sum(meta.label == label));
    end
end
