%% ===== User-tunable =====
SEG_HOURS            = 1;
TOTAL_HOURS          = 24 * 7;
dt                   = 0.002;
TARGET_PTS_PER_HOUR  = 1;
OOM_WARN_RATIO       = 1;

% テストモード
TEST_MODE            = true;
TEST_SIM_TIME        = 60;

% Pulse Generatorのパラメータ範囲
PULSE_AMP_MIN        = 0.1;
PULSE_AMP_MAX        = 5.0;
PULSE_PERIOD_MIN     = 0.1;
PULSE_PERIOD_MAX     = 10.0;
PULSE_WIDTH_MIN      = 10;
PULSE_WIDTH_MAX      = 90;
PULSE_PHASE_MIN      = 0;
PULSE_PHASE_MAX      = 1.0;

%% ===== 固定パラメータ =====
N_d        = 100;
N_for      = TOTAL_HOURS;
if TEST_MODE
    Sim_time = TEST_SIM_TIME;
    N_for = 2;
else
    Sim_time = SEG_HOURS * 3600;
end
f_low      = -1.0;
f_up       = 2.3;

%% ===== Simulink/QUARC setup =====
sim_filename = 'system_id_new';

fprintf('=== Setup Phase ===\n');
fprintf('Test Mode: %s\n', mat2str(TEST_MODE));
fprintf('Simulation Time: %d seconds\n', Sim_time);

try
    open_system(sim_filename);
    fprintf('Model opened\n');
catch ME
    error('Failed to open model %s: %s', sim_filename, ME.message);
end

% QUARC External Mode設定
try
    set_param(sim_filename, 'SimulationMode', 'external');
    fprintf('External Mode enabled\n');
catch ME
    warning('Could not set External Mode: %s', ME.message);
end

set_param(sim_filename, 'StopTime', num2str(Sim_time+10));

% Decimation設定（修正）
samples_per_hour = Sim_time / dt;
decim            = min(1, floor(samples_per_hour / TARGET_PTS_PER_HOUR));
fprintf('Decimation: %d\n', decim);

% To Workspace設定
toWksBlk = '';
try
    toWksBlk = [sim_filename, '/To Workspace'];
    get_param(toWksBlk, 'Handle');
    set_param(toWksBlk, 'Decimation', num2str(decim));
    try
        set_param(toWksBlk, 'SaveFormat', 'StructureWithTime');
    catch
    end
    fprintf('To Workspace configured\n');
catch
    try
        cands = find_system(sim_filename, 'LookUnderMasks', 'all', ...
                           'FollowLinks', 'on', 'BlockType', 'ToWorkspace');
        if ~isempty(cands)
            toWksBlk = cands{1};
            set_param(toWksBlk, 'Decimation', num2str(decim));
            try, set_param(toWksBlk, 'SaveFormat', 'StructureWithTime'); end
            fprintf('To Workspace found and configured\n');
        end
    catch
        warning('No To Workspace block');
    end
end

out_varname = 'simout';
if ~isempty(toWksBlk)
    try
        out_varname = get_param(toWksBlk, 'VariableName');
    catch
    end
end

% ログディレクトリ
logdir = fullfile(pwd, 'logs_hourly');
if ~exist(logdir, 'dir')
    mkdir(logdir);
    fprintf('Created log directory: %s\n', logdir);
end

% 初期化
gain_tuning = 20 / N_d;
frequency   = rand(N_d,1);
phase       = rand(size(frequency));
gain        = gain_tuning * rand(size(frequency));

% QUARC build
if exist('qc_build_model','file') == 2
    try
        fprintf('Building for QUARC...\n');
        qc_build_model;
        fprintf('Build complete\n');
    catch ME
        warning('qc_build_model failed: %s', ME.message);
    end
end

% To File block設定
toFileBlk = '';
try
    blk = [sim_filename, '/To File'];
    get_param(blk, 'Handle');
    toFileBlk = blk;
    fprintf('To File block found: %s\n', toFileBlk);
catch
    try
        tfCands = find_system(sim_filename, 'LookUnderMasks', 'all', ...
                             'FollowLinks', 'on', 'BlockType', 'ToFile');
        if ~isempty(tfCands)
            toFileBlk = tfCands{1};
            fprintf('To File block found: %s\n', toFileBlk);
        end
    catch ME
        warning('To File block not found: %s', ME.message);
    end
end

fprintf('\n=== Starting Main Loop ===\n\n');

%% ===== メインループ =====
t0 = datetime('now');

for k = 1:N_for
    fprintf('\n========== Run %d/%d ==========\n', k, N_for);
    
    try
        %% パラメータ生成
        freq_range = (f_up - f_low) * rand(N_d,1) + f_low;
        freq_range = sort(freq_range);
        frequency  = 10.^[f_low : (f_up - f_low)/N_d : f_up - (f_up - f_low)/N_d]';
        phase      = rand(size(frequency));
        gain       = gain_tuning * rand(size(frequency));
        
        pulse_amplitude = PULSE_AMP_MIN + (PULSE_AMP_MAX - PULSE_AMP_MIN) * rand();
        pulse_period    = PULSE_PERIOD_MIN + (PULSE_PERIOD_MAX - PULSE_PERIOD_MIN) * rand();
        pulse_width     = PULSE_WIDTH_MIN + (PULSE_WIDTH_MAX - PULSE_WIDTH_MIN) * rand();
        % pulse_phase     = PULSE_PHASE_MIN + (PULSE_PHASE_MAX - PULSE_PHASE_MIN) * rand();
        pulse_phase     =( PULSE_PHASE_MIN + PULSE_PHASE_MAX );
        % ワークスペースに設定
        assignin('base', 'frequency', frequency);
        assignin('base', 'phase', phase);
        assignin('base', 'gain', gain);
        assignin('base', 'pulse_amplitude', pulse_amplitude);
        assignin('base', 'pulse_period', pulse_period);
        assignin('base', 'pulse_width', pulse_width);
        assignin('base', 'pulse_phase', pulse_phase);
        
        fprintf('Pulse: Amp=%.3f, Period=%.3f s, Width=%.1f%%, Phase=%.3f s\n', ...
                pulse_amplitude, pulse_period, pulse_width, pulse_phase);
        
        %% To File設定（ファイル名のみ設定）
        ts_stamp       = datestr(datetime('now'), 'yyyymmdd_HHMMSS');
        input_mat_file = fullfile(logdir, sprintf('input_test_%s.mat', ts_stamp));
        
        if ~isempty(toFileBlk)
            try
                % ファイル名のみ設定（VariableNameは削除）
                set_param(toFileBlk, 'Filename', input_mat_file);
                fprintf('To File output: %s\n', input_mat_file);
            catch ME
                warning('Failed to configure To File: %s', ME.message);
            end
        end
        
        %% シミュレーション実行
        fprintf('Starting simulation (%d sec)...\n', Sim_time);
        tic;
        try
            if exist('qc_start_model','file') == 2
                qc_start_model;
            else
                set_param(sim_filename, 'SimulationCommand', 'start');
            end
            fprintf('Started (%.2f sec)\n', toc);
        catch ME
            error('Start failed: %s', ME.message);
        end
        
        % 待機（進捗表示付き）
        wait_time = Sim_time + 2;
        fprintf('Waiting %d seconds', wait_time);
        for wait_i = 1:wait_time
            pause(1);
            if mod(wait_i, 10) == 0
                fprintf('.');
            end
        end
        fprintf(' Done!\n');
        
        %% 停止
        fprintf('Stopping...\n');
        try
            if exist('qc_stop_model','file') == 2
                qc_stop_model;
            else
                set_param(sim_filename, 'SimulationCommand', 'stop');
            end
            pause(2);
            fprintf('Stopped\n');
        catch ME
            warning('Stop failed: %s', ME.message);
        end
        
        %% データ確認と保存
        fprintf('Checking saved data...\n');
        
        % To Fileで保存されたファイルを確認
        if exist(input_mat_file, 'file') == 2
            file_info = dir(input_mat_file);
            fprintf('  Input file exists: %s (%.2f KB)\n', input_mat_file, file_info.bytes/1024);
            
            % ファイルの中身を確認
            try
                input_data = load(input_mat_file);
                fprintf('  Variables in file: %s\n', strjoin(fieldnames(input_data), ', '));
                
                % 最初の変数を取得
                vars = fieldnames(input_data);
                if ~isempty(vars)
                    data = input_data.(vars{1});
                    if isnumeric(data)
                        fprintf('  Data size: %s\n', mat2str(size(data)));
                        fprintf('  Data range: [%.3f, %.3f]\n', min(data(:)), max(data(:)));
                    elseif isstruct(data)
                        fprintf('  Data is struct with fields: %s\n', strjoin(fieldnames(data), ', '));
                    elseif isa(data, 'timeseries')
                        fprintf('  Data is timeseries, Length: %d\n', length(data.Time));
                    end
                    input_ts = data;
                else
                    fprintf('  WARNING: File is empty!\n');
                    input_ts = [];
                end
            catch ME
                warning('Failed to load input file: %s', ME.message);
                input_ts = [];
            end
        else
            fprintf('  WARNING: Input file NOT created: %s\n', input_mat_file);
            input_ts = [];
        end
        
        %% 統合データ保存
        stamp   = datestr(datetime('now'), 'yyyymmdd_HHMMSS');
        matname = fullfile(logdir, sprintf('%s_IO_%s.mat', sim_filename, stamp));
        
        saved_vars = {};
        if evalin('base', sprintf('exist(''%s'',''var'')', out_varname))
            saved_vars{end+1} = out_varname;
        end
        
        candidates = {'u','in','input','in_data','y','out','output','out_data','logsout','simout'};
        for ci = 1:numel(candidates)
            vn = candidates{ci};
            if evalin('base', sprintf('exist(''%s'',''var'')', vn))
                saved_vars{end+1} = vn;
            end
        end
        saved_vars = unique(saved_vars, 'stable');
        
        if ~isempty(saved_vars)
            fprintf('  Output variables: %s\n', strjoin(saved_vars, ', '));
        else
            fprintf('  WARNING: No output variables found in workspace\n');
        end
        
        % 保存
        try
            if isempty(saved_vars)
                save(matname, 'input_ts', 'frequency', 'phase', 'gain', ...
                     'pulse_amplitude', 'pulse_period', 'pulse_width', 'pulse_phase', '-v7.3');
            else
                tmpCell = [saved_vars, {'frequency','phase','gain','input_ts', ...
                           'pulse_amplitude','pulse_period','pulse_width','pulse_phase'}];
                tmpCell = unique(tmpCell, 'stable');
                for iVar = 1:numel(saved_vars)
                    vn = saved_vars{iVar};
                    assignin('caller', vn, evalin('base', vn));
                end
                save(matname, tmpCell{:}, '-v7.3');
            end
            fprintf('  Saved: %s\n', matname);
        catch ME
            warning('Save failed: %s', ME.message);
        end
        
        % クリーンアップ
        for vi = 1:numel(saved_vars)
            try, evalin('base', sprintf('clear %s', saved_vars{vi})); end
        end
        clear input_ts
        
        fprintf('Run %d complete\n', k);
        
    catch ME
        fprintf('\nERROR in run %d: %s\n', k, ME.message);
        if ~isempty(ME.stack)
            fprintf('Location: %s (line %d)\n', ME.stack(1).name, ME.stack(1).line);
        end
        
        % 緊急停止
        try
            if exist('qc_stop_model','file') == 2
                qc_stop_model;
            else
                set_param(sim_filename, 'SimulationCommand', 'stop');
            end
        catch
        end
        
        rethrow(ME);
    end
    
    if k >= TOTAL_HOURS || datetime('now') - t0 >= days(7)
        break;
    end
end

fprintf('\n=== Complete ===\n');
fprintf('Logs in: %s\n', logdir);

