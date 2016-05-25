function concat_ftrs
%concat_ftrs
%loads features from all ftr.mat files in a directory, concatenates them
%into one cell array, saves that array 

% Create batchfile with all .not.mat files
if isunix
    !ls *.not.mat > batchfile
else
    !dir /B **.not.mat > batchfile
end

fid=fopen('batchfile','r');

CAT_labels = [];
CAT_syl_durations = [];
CAT_pre_durations = [];
CAT_foll_durations = [];
CAT_pre_gapdurs = [];
CAT_foll_gapdurs = [];
CAT_mn_amp_smooth_rect = [];
CAT_mn_amp_rms = [];
CAT_mn_spect_entropy = [];
CAT_mn_hi_lo_ratio = [];
CAT_delta_amp_smooth_rect = [];
CAT_delta_entropy = [];
CAT_delta_hi_lo_ratio = [];

counter = 1; % used for song ID vector
song_IDs = [];
%song ID vector makes it possible to determine where songs start/end even
%after concatenating and removing unwanted samples (e.g., weird syllables,
%calls that shouldn't go into training set)
%need to identify where songs start/stop for bootstrap analyses

% Loop throu all .not.mat files in current directory
while 1
    notmat_fn=fgetl(fid);
    ftr_fn = [notmat_fn(1:end-8) '.ftr.mat'];
    % Break while loop when last filename is reached
    if (~ischar(notmat_fn));break;end

    disp(['loading: ' ftr_fn])
    load(notmat_fn,'labels')
    load(ftr_fn)
    
    CAT_labels = [CAT_labels labels];
    CAT_syl_durations = [CAT_syl_durations;syl_durations];
    CAT_pre_durations = [CAT_pre_durations;pre_durations];
    CAT_foll_durations = [CAT_foll_durations;foll_durations];
    CAT_pre_gapdurs = [CAT_pre_gapdurs;pre_gapdurs];
    CAT_foll_gapdurs = [CAT_foll_gapdurs;foll_gapdurs];
    CAT_mn_amp_smooth_rect = [CAT_mn_amp_smooth_rect;mn_amp_smooth_rect];
    CAT_mn_amp_rms = [CAT_mn_amp_rms;mn_amp_rms];
    CAT_mn_spect_entropy = [CAT_mn_spect_entropy;mn_spect_entropy];
    CAT_mn_hi_lo_ratio = [CAT_mn_hi_lo_ratio;mn_hi_lo_ratio];
    
    if exist('delta_amp_smooth_rect_array','var')
        CAT_delta_amp_smooth_rect = [CAT_delta_amp_smooth_rect;delta_amp_smooth_rect_array];
        CAT_delta_entropy = [CAT_delta_entropy;delta_entropy_array];
        CAT_delta_hi_lo_ratio = [CAT_delta_hi_lo_ratio;delta_hi_lo_ratio_array];
    end
    
    song_ID_tmp = ones(size(labels)) * counter;
    song_ID_tmp = song_ID_tmp'; % transpose to column vector
    song_IDs = [song_IDs;song_ID_tmp];
    counter = counter + 1;
end % of while 1 loop

NUM_FEATURES = 9; 
NUM_FEATURES_PLUS_DELTAS = 12;

if ~isempty(CAT_delta_amp_smooth_rect)
    feature_cell = cell(2,NUM_FEATURES_PLUS_DELTAS);
else
    feature_cell = cell(2,NUM_FEATURES);
end

feature_cell{1,1} = 'syl_durations';feature_cell{2,1} = CAT_syl_durations;
feature_cell{1,2} = 'pre_durations';feature_cell{2,2} = CAT_pre_durations;
feature_cell{1,3} = 'foll_durations';feature_cell{2,3} = CAT_foll_durations;
feature_cell{1,4} = 'pre_gapdurs';feature_cell{2,4} = CAT_pre_gapdurs;
feature_cell{1,5} = 'foll_gapdurs';feature_cell{2,5} = CAT_foll_gapdurs;
feature_cell{1,6} = 'mn_amp_smooth_rect';feature_cell{2,6} = CAT_mn_amp_smooth_rect;
feature_cell{1,7} = 'mn_amp_rms';feature_cell{2,7} = CAT_mn_amp_rms;
feature_cell{1,8} = 'mn_spect_entropy';feature_cell{2,8} = CAT_mn_spect_entropy;
feature_cell{1,9} = 'mn_hi_lo_ratio';feature_cell{2,9} = CAT_mn_hi_lo_ratio;

if ~isempty(CAT_delta_amp_smooth_rect)
    feature_cell{1,10} = 'delta_amp_smooth_rect';feature_cell{2,10} = CAT_delta_amp_smooth_rect;
    feature_cell{1,11} = 'delta_entropy';feature_cell{2,11} = CAT_delta_entropy;
    feature_cell{1,12} = 'delta_hi_lo_ratio';feature_cell{2,12} = CAT_delta_hi_lo_ratio;
end

labels = CAT_labels;

% get filename of first cbin
dir = ls('*.cbin');
a_cbin = dir(1,:);
pat = '[a-z]{2}\d{1,3}[a-z]{2}\d{1,3}';
birdname = char(regexp(a_cbin,pat,'match')); % use regexp to extract birdname

dir_datenum = fn2datenum(a_cbin); % if 1, fn2datenum returns just datenum corresponding mmddyyyy 00:00:00
dir_datestr = datestr(dir_datenum,'mmddyyyy');
now_datestr = datestr(now,'mmddyyyy');
save_fname = [birdname '_ftr_cell_' dir_datestr '_generated_' now_datestr];
disp(['saving: ' save_fname]);
save(save_fname,'feature_cell','labels','song_IDs')