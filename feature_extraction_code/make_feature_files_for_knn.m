function make_feature_files(quantify_deltas,t_early,t_late)
% make_feature_files
%   extracts features from spectrograms in .spect.mat files
%   and saves in .ftr.mat files
% 
%   input arguments:
%       quantify_deltas -- if quantify_deltas = 1, then take the difference of
%           the values of the extracted features at the times t_early and 
%           t_late. Default is quantify_deltas = 0 (i.e., don't quantify)
%       t_early -- "early" time in percent * 0.01. Default is 0.2 (i.e., 20%)
%       t_late -- "late" time in percent * 0.01. Default is 0.8

if nargin<1
    quantify_deltas = 0;
end

if nargin<2
    t_early = 0.2;
end

if nargin<3
    t_late = 0.8;
end


% Create batchfile with all .not.mat files
if isunix;
    !ls *.not.mat > batchfile
else
    !dir /B **.not.mat > batchfile
end

fid=fopen('batchfile','r');

% Loop throu all .not.mat files in current directory
while 1
    notmat_fn=fgetl(fid); % get name of .not.mat filename
    spect_fn = ...
        [notmat_fn(1:end-8) '.spect.mat'];
    % Break while loop when last filename is reached
    if (~ischar(notmat_fn));break;end

    load(notmat_fn,'onsets','offsets');
    load(spect_fn)
    
    syl_durations = offsets-onsets;
    pre_durations = [0;syl_durations(1:end-1)];
    foll_durations = [syl_durations(2:end);0];
    gapdurs = onsets(2:end) - offsets(1:end-1);
    pre_gapdurs = [0;gapdurs];
    foll_gapdurs = [gapdurs;0];

    num_syls = length(onsets);
    
    amp_smooth_rect_cell = cell(num_syls,1);
    amp_rms_cell = cell(num_syls,1);
    %wiener_entropy_cell = cell(num_syls,1);
    spectral_entropy_cell = cell(num_syls,1);
    hi_lo_ratio_cell = cell(num_syls,1);

    if quantify_deltas
        delta_amp_smooth_rect_array = [];
        delta_entropy_array = [];
        delta_hi_lo_ratio_array = [];
    end
    
    %extract features from each spectral slice for each syllable
    for syl_id=1:length(onsets)
        raw_syl = rawsyl_cell{syl_id};
        [amp_sm_rect,amp_rms] = compute_amps(raw_syl,Fs,win_duration,overlap);
        
        syl_spect = spect_cell{syl_id};
        syl_psd = psd_cell{syl_id};
        syl_freqbins = freqbins_cell{syl_id};
        syl_timebins = timebins_cell{syl_id};
        spectral_entropy = [];
        hi_lo_ratio = [];
        % wiener_entropy = [];
        
        for slice_id = 1:size(syl_spect,2)
            spect_slice = syl_spect(:,slice_id);
            % make spectrum into power spectral density
            psd = (abs(spect_slice)).^2; % changed
            % normalize psd --> probability density function (pdf)
            psd_pdf = psd / sum(psd);
            spectral_entropy = [spectral_entropy -sum(psd_pdf .* log(psd_pdf))];
            
            %
            scalar_hi_lo_ratio = log10(sum(psd(syl_freqbins<5000)) ./ sum(psd(syl_freqbins>5000)));
            hi_lo_ratio = [hi_lo_ratio scalar_hi_lo_ratio];
        end

        
        if quantify_deltas
            t_early_in_s = t_early * (syl_durations(syl_id)/1000);
            t_early_id = find(abs((syl_timebins - t_early_in_s)) == min(abs(syl_timebins - t_early_in_s)));
            %if "early" time to quantify is halfway between two time bins,
            %choose the earliest of the two
            if length(t_early_id)>1
                t_early_id = min(t_early_id);
            end
            
            t_late_in_s = t_late * (syl_durations(syl_id)/1000);
            t_late_id = find(abs((syl_timebins - t_late_in_s)) == min(abs(syl_timebins - t_late_in_s)));
            %likewise, if "late" time to quantify is halfway between two time
            %bins, choose the latest
            if length(t_late_id)>1
                t_late_id = min(t_late_id);
            end
            
            
            delta_amp_smooth_rect =  amp_sm_rect(t_late_id) - amp_sm_rect(t_early_id);
            delta_amp_smooth_rect_array = [delta_amp_smooth_rect_array;delta_amp_smooth_rect];
            delta_entropy = spectral_entropy(t_late_id) - spectral_entropy(t_early_id);
            delta_entropy_array = [delta_entropy_array;delta_entropy];
            delta_hi_lo_ratio = hi_lo_ratio(t_late_id) - hi_lo_ratio(t_early_id);
            delta_hi_lo_ratio_array = [delta_hi_lo_ratio_array;delta_hi_lo_ratio];
        end
        
        amp_smooth_rect_cell{syl_id} = amp_sm_rect;
        amp_rms_cell{syl_id} = amp_rms;
        spectral_entropy_cell{syl_id} = spectral_entropy;
        hi_lo_ratio_cell{syl_id} = hi_lo_ratio;

       
    end % of loop to loop through syllables

    mn_amp_smooth_rect = cellfun(@mean,amp_smooth_rect_cell);
    mn_amp_rms = cellfun(@mean,amp_rms_cell);
    mn_spect_entropy = cellfun(@mean,spectral_entropy_cell);
    mn_hi_lo_ratio = cellfun(@mean,hi_lo_ratio_cell);
    
    %list of variables to save -- note that you have to have a space
    %after each variable name so they concatenate as one long string
    %with each name separated by a space
    save_vars = ['syl_durations ',...
        'pre_durations ',...
        'foll_durations ',...
        'pre_gapdurs ',...
        'foll_gapdurs ',...
        'amp_smooth_rect_cell ',...
        'amp_rms_cell ',...
        'spectral_entropy_cell ',...
        'hi_lo_ratio_cell ',...
        'mn_amp_smooth_rect ',...
        'mn_amp_rms ',...
        'mn_spect_entropy ',...
        'mn_hi_lo_ratio '];
    
    if quantify_deltas
        save_vars = [save_vars,...
            'delta_amp_smooth_rect_array ',...
            'delta_entropy_array ',...
            'delta_hi_lo_ratio_array'];
    end
    
    ftr_fn = [notmat_fn(1:end-8) '.ftr.mat'];
    disp(['Saving: ' ftr_fn])
    eval(sprintf('save %s %s',ftr_fn,save_vars))
    
end
