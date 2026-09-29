function plot_examples()
%PLOT_EXAMPLES  Plot one example run per class and save a PNG.
%
%   Simulates the nominal motor three times (Healthy, High Load, Increased
%   Friction) with the SAME fault onset (4 s) and severity (2x), so the effect
%   of each fault is easy to compare. Saves results/matlab_example_signals.png.

    rng(0);
    p           = motor_params();
    fault_onset = 4;      % [s]
    severity    = 2.0;
    class_names = ["Healthy", "High Load", "Increased Friction"];
    % Colorblind-safe palette (same colors as the Python plots / dashboard)
    colors      = [0.165 0.471 0.839;   % blue   #2a78d6 - healthy
                   0.922 0.408 0.204;   % orange #eb6834 - high load
                   0.106 0.686 0.478];  % aqua   #1baf7a - friction

    runs = cell(1, 3);
    for label = 0:2
        runs{label+1} = simulate_motor(p, label, severity, fault_onset);
    end

    signals = ["current" "speed_rpm" "torque" "temperature"];
    ylabels = ["Current [A]" "Speed [RPM]" "Torque [N·m]" "Temperature [°C]"];

    % Show the window when run from the MATLAB desktop; stay hidden in
    % terminal (-batch) runs, where there is no screen to draw on.
    interactive = usejava('desktop');
    fig = figure('Visible', interactive, 'Position', [100 100 1100 750]);
    if exist('theme', 'file')  % theme() exists from R2025a; older releases are light already
        theme(fig, 'light');   % force a light figure even if the desktop is in dark mode
    end
    tl  = tiledlayout(fig, 2, 2, 'TileSpacing', 'compact', 'Padding', 'compact');
    for s = 1:numel(signals)
        ax = nexttile(tl);
        hold(ax, 'on');
        for c = 1:3
            plot(ax, runs{c}.time, runs{c}.(signals(s)), 'LineWidth', 1.6, ...
                 'Color', colors(c, :), 'DisplayName', class_names(c));
        end
        xline(ax, fault_onset, '--k', 'fault onset', 'LineWidth', 1.2, ...
              'LabelVerticalAlignment', 'middle', 'HandleVisibility', 'off');
        grid(ax, 'on');
        xlabel(ax, 'Time [s]');
        ylabel(ax, ylabels(s));
        if s == 1
            legend(ax, 'Location', 'best');
        end
    end
    title(tl, sprintf('DC motor digital twin: one run per class (severity %.1fx)', severity));

    root    = fileparts(fileparts(mfilename('fullpath')));
    out_dir = fullfile(root, 'results');
    if ~exist(out_dir, 'dir'); mkdir(out_dir); end
    out_path = fullfile(out_dir, 'matlab_example_signals.png');
    exportgraphics(fig, out_path, 'Resolution', 150);
    if ~interactive
        close(fig);   % keep the window open for the user when interactive
    end
    fprintf('Saved %s\n', out_path);
end
