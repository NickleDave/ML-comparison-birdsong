!dir /B *.not.mat > batchfile
n_files = size(ls('*.not.mat'),1); % size(ls,rows) = number of .not.mat files

% initialization
ct_files = 0 ; % counter to keep track of files analyzed
ct=1; % counter for number of syllables
song_counter = 1; % used for song ID vector
song_IDs_vec = [];
%song ID vector makes it possible to determine where songs start/end even
%after concatenating and removing unwanted samples (e.g., weird syllables,
%calls that shouldn't go into training set)
%need to identify where songs start/stop for bootstrap analyses

features_mat = {}; %"declare" cell array so interpreter doesn't gripe and crash

fid=fopen('batchfile','r');
while 1
    fn=fgetl(fid);
    if (~ischar(fn));break;end
    cbin_fn=fn(1:end-8);
 
    load(fn);
    onsets=onsets/1000; % convert back to s
    offsets=offsets/1000; % convert back to s
    
    %additional features--testing whether these improve SVM accuracy
    syl_durations = offsets-onsets;
    pre_durations = [0;syl_durations(1:end-1)];
    foll_durations = [syl_durations(2:end);0];
    gapdurs = onsets(2:end) - offsets(1:end-1);
    pre_gapdurs = [0;gapdurs];
    foll_gapdurs = [gapdurs;0];

    [rawsong, Fs] = evsoundin('.', cbin_fn,'obs0');

    %%% go through syllable by syllable and quantify different parameters %%%
    t = (1:length(rawsong)) / Fs;
    for x = 1:length(onsets)
        on = onsets(x);
        off = offsets(x);
        on_id = find(abs((t-on))==min(abs(t-on)));
        off_id = find(abs((t-off))==min(abs(t-off)));
        syl_wav = rawsong(on_id:off_id);
        % get features from syl using Tachibana's function
        features_vec = makeAllFeatures(syl_wav,Fs);
        % add duration features to those returned by Tachibana func
        dur_features = [pre_durations(x) foll_durations(x) pre_gapdurs(x) foll_gapdurs(x)];
        %appending to a cell gives you a growing row "vector" (of cells)
        %Why this is important is explained in line 52 below
        features_mat{ct} = [features_vec dur_features];
        label_vec(ct) = double(labels(x));
        ct = ct + 1;
    end
    
    song_ID_tmp = ones(size(labels)) * song_counter;
    song_ID_tmp = song_ID_tmp'; % transpose to column vector
    song_IDs_vec = [song_IDs_vec;song_ID_tmp];
    song_counter = song_counter + 1;
    
    ct_files = ct_files + 1;
    disp(['Processed ' num2str(ct_files) ' of ' num2str(n_files) ' files: ' fn])
end

%Why it's important that features_mat is a row vector:
%If you run cell2mat on it, you get one really long row
%when what you actually want is each cell to be a row, with each element in
%each cell being a column. At least that's what I want here.
features_mat = cell2mat(features_mat'); % <--Hence the transpose

d_num=fn2datenum(cbin_fn); % get date from filename
dstr = datestr(d_num,'mm-dd-yy');
underscore_ids = strfind(cbin_fn,'_');
bird_name = cbin_fn(1:(underscore_ids(1)-1));
save_fname = [bird_name '_feature_file_from_' dstr '_generated_' datestr(now,'mm-dd-yy_HH-MM')];
save(save_fname,'features_mat','label_vec','song_IDs_vec')