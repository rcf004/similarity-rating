function awk_vids(subj_id, moviename_num)
% Screen('Preference', 'SkipSyncTests', 1);

AssertOpenGL; %InitializeMatlabOpenGL(1);
Screen('CloseAll')

clear w1 w2 ftimes subj_responses movie gamepad* outfile

ftimes = [];


%% temporary, but extremely important:
v = bitor(2^16, Screen('Preference','ConserveVRAM'));
Screen('Preference','ConserveVRAM', v);

rand('twister',sum(100*clock));

addpath(genpath('/Users/nimblelab/Desktop/fmri2022/generic_routines'))

%Flags
debugging_mode = 0; 
eyetracking = 0;
visualize_flag = 0;
infant_mode = 0;
gamepad = 1;
bar_feedback = 1;


if infant_mode, 
    pos = [0.1 0.1;
        0.9 0.1;
        0.5 0.5;
        0.1 0.9;
        0.9 0.9;];
else
    pos = [0.1 0.1;
        0.5 0.1;
        0.9 0.1;
        0.1 0.5;
        0.5 0.5;
        0.9 0.5;
        0.1 0.9;
        0.5 0.9;
        0.9 0.9;];
end


subj_id = input('Please enter subject ID:  ','s');

subj_responses = [];

keyIsDown_ext_prev = 0;

%% Show checkboard and test



%%
disp('Please enter clip:');
disp('#1 For the Petting Zoo:');
disp('#2 For the Souvenir Shop:');
disp('#3 For the Checkboard:');

moviename_num = input(':  ');
if moviename_num == 1
    moviename = [pwd filesep 'NathanForYou/Full_shows/NFY_PettingZoo' num2str(moviename_num) '.mp4'];
    task_str = ['ClipZoo_' num2str(moviename_num)];
elseif moviename_num == 2
   moviename = [pwd filesep 'NathanForYou/Full_shows/NFY_SouvenirShop' num2str(moviename_num) '.mp4'];
%    moviename = [pwd filesep 'NathanForYou/Full_shows/old_NFY_SouvenirShop.mp4'];   
    task_str = ['ClipShop_' num2str(moviename_num)];
elseif moviename_num == 3
    moviename = [pwd filesep 'NathanForYou/Full_shows/control.mp4'];
    task_str = ['ClipControl_' num2str(moviename_num)];    
end

%%
outdir1 = ['out_data/' subj_id '/'];
outfile = [subj_id '_awk_' task_str '_' datestr(now,30) '.mat'];
mkdir(outdir1);

% outdir2 = ['/Users/akrendl/Desktop/fmri2022/soc_awk/' task_str filesep];
% mkdir(outdir2);

% % %Check to see if output file already exists
if exist([outdir1 outfile],'file')
    ButtonName = questdlg('WARNING: File Exists. Continuing will overwrite old data. Continue anyway?',...
        'Log File Exists',...
        'Yes','No','No');
    switch ButtonName
        case 'No'
            return;
    end
end

% if exist([outdir2 outfile],'file')
%     ButtonName = questdlg('WARNING: File Exists. Continuing will overwrite old data. Continue anyway?',...
%         'Log File Exists',...
%         'Yes','No','No');
%     switch ButtonName
%         case 'No'
%             return;
%     end
% end


%% keyboard setup:

% for scanner trigger
[extKeyboard,intKeyboard] = tobii_find_keyboard;

KbName('UnifyKeyNames');
keys.stopkey = 'q';
keys.trigger = 'space';
keys.pausekey = 'p';
keys.subj_response = 'space';

if ~debugging_mode,
        ListenChar(2);
    HideCursor;
end

% HideCursor % remeber to uncomment

%% GamePad setup:

joystick_range = 512;
error_percent = 3;

if gamepad,
    gamepad_out = NaN(16000,2);
    gamepad_axis = 2; %up/down
    
    gamepad_name = Gamepad('GetGamepadNamesFromIndices', 1);
    gamepad_index = Gamepad('GetGamepadIndicesFromNames', gamepad_name);
    handles = Gamepad('GetAxisRawMapping', gamepad_index, gamepad_axis);
    
end
%%

% % % for testing on single screen:
% % [w1.window] = Screen('Openwindow', 0, [128 128 128],[0 0 100 100]);
% % [w2.window] =  

Screen('Openwindow', 0, [128 128 128],[120 120 220 220]);

% 
if debugging_mode, 
% [w1] = initialize_screen(0,[0 0 0],[0 0 1920 1080]);
[w1] = initialize_screen(0,[0 0 0],[0 0 800 800]);
else
[w1] = initialize_screen(0,[0 0 0],[0 0 2650 1440]);
% [w1] = initialize_screen(0,[0 0 0],[0 0 1920 1080]);
end
w1.datetime = clock;
% w2.datetime = clock;

% res_scaling_x = w2.res(3)/w1.res(3);
% res_scaling_y = w2.res(4)/w1.res(4);

res_scaling_x = 1;
res_scaling_y = 1;


%% for movie event marking:
increment = 1;
marker_count = 1;

%%

[movie duration fps mov_width mov_height] = Screen('OpenMovie', w1.window, moviename);

%WARNING: HARD CODED:
mov_width = 1538;
mov_height = 865;

total_time_temp = 0;
if total_time_temp ~=0,
    movie_onset_time = Screen('SetMovieTimeIndex', movie, total_time_index);
end


timeindex = [];
paused_time = 0;
manual_stop_time = inf;


%% movie size

border_ratio = .1;
if moviename_num == 3
    movie_scaling_factor = (w1.ScreenWidth/mov_width - border_ratio) / 2;
else
    movie_scaling_factor = (w1.ScreenWidth/mov_width - border_ratio) / 1.2;
end    
% movie_scaling_factor = 1;
w1.mov_coords = [w1.ScreenCenterX - mov_width/2*movie_scaling_factor w1.ScreenCenterY - mov_height/2*movie_scaling_factor w1.ScreenCenterX + mov_width/2*movie_scaling_factor w1.ScreenCenterY + mov_height/2*movie_scaling_factor];
% w2.mov_coords = [w2.ScreenCenterX - mov_width*res_scaling_x/2*movie_scaling_factor w2.ScreenCenterY - mov_height*res_scaling_y/2*movie_scaling_factor w2.ScreenCenterX + mov_width*res_scaling_x/2*movie_scaling_factor w2.ScreenCenterY + mov_height*res_scaling_y/2*movie_scaling_factor];

%% bar size

bar_width = 20;
bar_offset = 5;

w1.bar_coords = [w1.mov_coords(3)+bar_offset w1.mov_coords(2) w1.mov_coords(3)+bar_offset + bar_width w1.mov_coords(4) ];
% w2.bar_coords = [w2.mov_coords(3)+bar_offset w2.mov_coords(2) w2.mov_coords(3)+bar_offset + bar_width w2.mov_coords(4) ];

w1.bar_range = w1.bar_coords(4) - w1.bar_coords(2);
% w2.bar_range = w2.bar_coords(4) - w2.bar_coords(2);

w1.units_scaled = w1.bar_range/joystick_range;
% w2.units_scaled = w2.bar_range/joystick_range;




%% write some info into w1 and w2 structures:

w1.duration = duration;
w1.fps = fps;
w1.mov_width = mov_width;
w1.mov_height = mov_height;
w1.border_ratio = border_ratio;
w1.movie_scaling_factor = movie_scaling_factor;

% w2.duration = duration;
% w2.fps = fps;
% w2.mov_width = mov_width;
% w2.mov_height = mov_height;
% w2.border_ratio = border_ratio;
% w2.movie_scaling_factor = movie_scaling_factor;

%%

% if moviename_num < 3 % == 5
%     [quit_flag] = WaitForScannerTrigger(w1,debugging_mode,keys,intKeyboard,extKeyboard,'Push the joystick whenever you feel awkward or socially uncomfortable',30); %subfunction
% else


myText1 = ['You will now watch a short video in which an image on the screen will change from lighter to darker.\n' ...
        'We want to know if and when you think the image is getting darker overall. You will push the joystick\n' ...
        'forward to the extent that you think the image is getting darker.When you do not push the joystick at all,\n' ...
        'this indicates that the image is light. When you push the joystick all the way forward,\n' ... 
        'this would indicate that the image as dark as it can be. If you find the image is getting dark,\n' ... 
        'you should continue pressing the joystick forward to whatever extent you this is appropriate for as long as it remains dark.\n' ... 
        'Do the best you can. There are no right or wrong answers.\n'];
    
myText2 = ['You will now watch a short video. We want to know if and when you find any of the events in the video to be socially awkward.\n' ...
        'Some events may not be awkward at all, others might be moderately awkward, and others might be very awkward. You will push the joystick forward\n' ...
        'to the extent that you think what is happening in the scene is awkward. When you do not push the joystick at all, this indicates that the scene is ‘not at all awkward.\n' ...
        'When you push the joystick all the way forward, this would indicate the highest level of awkwardness. If you find the scene awkward to any degree,\n' ...
        'you should continue pressing the joystick forward to whatever extent you this is appropriate for as long as the scene continued to be awkward.\n' ...
        'Do the best you can. There are no right or wrong answers.\n'];    
    
% Draw 'myText', centered in the display window:
if moviename_num == 3
    DrawFormattedText(w1.window, myText1, 'center', 'center', [128 128 128]);
    Screen('Flip', w1.window);
elseif moviename_num ~=3
    DrawFormattedText(w1.window, myText2, 'center', 'center', [128 128 128]);
    Screen('Flip', w1.window);
end
    
while 1,
    [ keyIsDown, timeSecs, keyCode ] = KbCheck(intKeyboard);
    if strcmpi(KbName(keyCode),'space')
        break
    end
end



    [quit_flag] = WaitForScannerTrigger(w1,debugging_mode,keys,intKeyboard,extKeyboard,'The video will begin shortly...',30); %subfunction
% end
% 
% Beeper(1000,1,1);

trigger_time = GetSecs;

if quit_flag,
    Screen('CloseMovie', movie);
    disp('Experiment manually aborted.');
    ShowCursor;
%     ListenChar(0);
    clear screen
    return;
end

% WaitSecs(1);

if eyetracking
    talk2tobii('START_TRACKING');
    WaitSecs(1);
    talk2tobii('RECORD');
    talk2tobii('EVENT','initializing_call',1,'time',GetSecs);
end

%%

% movie_onset_time = Screen('SetMovieTimeIndex', movie, 2);
aaa = GetSecs;

if  strcmpi(task_str,'joystick_test'),
    Screen('PlayMovie', movie, 1, 0, 0.2); %start
else
    Screen('PlayMovie', movie, 1, 0, 1.0); %start
end

% timeindex= Screen('GetMovieTimeIndex', movie);
% timeindex_old = timeindex;
% while timeindex == timeindex_old, %i.e., while frames haven't started appearing...
%     timeindex = Screen('GetMovieTimeIndex', movie)
%     WaitSecs(.1);
% end

% timeindex= Screen('GetMovieTimeIndex', movie)
% timeindex_old = timeindex;
% while timeindex == timeindex_old, %i.e., while frames haven't started appearing...
%     timeindex = Screen('GetMovieTimeIndex', movie)
%     WaitSecs(.1)
% end


% WaitSecs(2)

start_time = GetSecs;
if eyetracking,
    talk2tobii('EVENT','start_movie',start_time);
end


w1.start_gap = start_time - trigger_time;

rawState = -inf;
joystick_count = 1;


%%
% return
%%

start_flag = 0; 
% counter = 0;
while 1,
    

    
    
    [frame_tex, movie_time] = Screen('GetMovieImage', w1.window,movie,0);

%     disp(num2str(frame_tex));
    %%%%NEW DPK
    if frame_tex == 0
        % No new frame in polling wait (blocking == 0). Just sleep
        % a bit and then retry.
        WaitSecs(0.001);
        continue;
    end
    %%%%
    
    
%         disp(num2str(movie_time));

    if frame_tex > 0,
        
    if start_flag == 0,
        start_delay = GetSecs - start_time;
        start_flag  = 1;
    end

%         counter = counter+ 1;
%         if counter == 100, 
%             out = GetSecs - start_time
%             clear screen
%             return;
%         end
        
        if gamepad,
            rawState = PsychHID('RawState', handles(1), handles(2));
            
            if rawState > joystick_range, 
                rawState = joystick_range;
            end
            
            rawState = abs(rawState - joystick_range);
            
            %ignore responses less than 3% total range, as this corresponds
            %mostly to noise.
            if rawState <= round((joystick_range/100)*error_percent),
                rawState = round((joystick_range/100)*error_percent);
            end
%             disp(num2str(rawState));
            gamepad_out(joystick_count,:) = [rawState GetSecs];
            joystick_count = joystick_count + 1;
            
        else
            rawState = rawState + 1;

            joystick_count = joystick_count + 1;

            if rawState > joystick_range,
                rawState = joystick_range;
            end
            if rawState <= round((joystick_range/100)*error_percent),
                rawState = round((joystick_range/100)*error_percent);
            end
            gamepad_out(joystick_count,:) = [rawState GetSecs];

        end
        
        if bar_feedback, 
            Screen('FrameRect', w1.window, [255 255 255], w1.bar_coords, 1);
%             Screen('FrameRect', w2.window, [255 255 255], w2.bar_coords, 1);
            
            y_value_1 = rawState * w1.units_scaled;
%             y_value_2 = rawState * w2.units_scaled;
            
            Screen('FillRect', w1.window, [0 0 200], [w1.bar_coords(1)+1 w1.bar_coords(4)+1-y_value_1 w1.bar_coords(3)-1 w1.bar_coords(4)-1]);
%             Screen('FillRect', w2.window, [0 0 200], [w2.bar_coords(1)+1 w2.bar_coords(4)+1-y_value_2 w2.bar_coords(3)-1 w2.bar_coords(4)-1]);
        end
        
        if eyetracking,
            if movie_time >= increment*marker_count,
%                 talk2tobii('EVENT','movie_time',increment*marker_count);
                talk2tobii('EVENT','movie_time',increment*marker_count,'GetSecs',GetSecs);
                marker_count = marker_count+1;
            end
        end
        
        
%         Screen('DrawTexture', w1.window, frame_tex);
%         Screen('DrawTexture', w2.window, frame_tex);
        
        %Important: must scale movie dimentions for alignment
        Screen('DrawTexture', w1.window, frame_tex,[],w1.mov_coords);
%         Screen('DrawTexture', w2.window, frame_tex,[],w2.mov_coords);
        
        
        %%
        if visualize_flag == 1,
            
            
            x_L = -1;
            x_R = -1;
            y_L = -1;
            y_R = -1;
            
            if ~eyetracking,
                [x_L_temp, y_L_temp, valid_data] = GetMouse;
                x_L = x_L_temp/w1.res(3);
                y_L = y_L_temp/w1.res(4);
                x_R = x_L;
                y_R = y_L;
                LValid = sum(valid_data);
                RValid = sum(valid_data);
                LDist = 0;
                RDist = 0;
                
            elseif eyetracking,
                
                gazeData=talk2tobii('GET_SAMPLE_EXT');
                
                LValid = gazeData(7);
                if LValid <= 1,
                    x_L = gazeData(1);
                    y_L = gazeData(2);
                end
                
                RValid = gazeData(8);
                if RValid <= 1,
                    x_R = gazeData(3);
                    y_R = gazeData(4);
                end
                
                LDist = round(gazeData(13))/10;
                RDist = round(gazeData(14))/10;
                
            end
            
            if LValid >= 1,
                Screen('DrawDots',w2.win,[w2.res(3)-60 50],40,[250 0 0],[0 0],2);
                if debugging_mode
                    Screen('DrawDots',w1.win,[w1.res(3)-60 50],40,[250 0 0],[0 0],2);
                end
            end
            
            if RValid >= 1,
                Screen('DrawDots',w2.win,[w2.res(3)-30 50],40,[250 0 0],[0 0],2);
                if debugging_mode
                    Screen('DrawDots',w1.win,[w1.res(3)-30 50],40,[250 0 0],[0 0],2);
                end
            end
            
            
            %Disp Distance to left and right eye:
            Screen('TextSize', w2.win, 50);
            Screen('DrawText', w2.win, ['LDist: ' num2str(LDist)], 50, 80, [255 255 0]);
            Screen('DrawText', w2.win, ['RDist: ' num2str(RDist)], 50, 140, [255 255 0]);
            Screen('DrawText', w2.win, ['Time Remaining: '...
                num2str(floor(round(duration - (GetSecs-start_time - paused_time))/60)) ' min '...
                num2str(round(duration - (GetSecs-start_time-paused_time))-(60*(floor(round(duration - (GetSecs-start_time-paused_time))/60)))) ' sec'],...
                400, 80, [255 255 0]);
                            
            
            %draw gaze/mouse location
            if debugging_mode
                Screen('DrawDots',w1.win,round([x_L*w1.res(3) y_L*w1.res(4)]),40,[0 0 200],[0 0],2);
                Screen('DrawDots',w1.win,round([x_R*w1.res(3) y_R*w1.res(4)]),40,[0 200 0],[0 0],2);
            end
            
            Screen('DrawDots',w2.win,round([x_L*w2.res(3) y_L*w2.res(4)]),40,[0 0 200],[0 0],2);
            Screen('DrawDots',w2.win,round([x_R*w2.res(3) y_R*w2.res(4)]),40,[0 200 0],[0 0],2);
            
            
        end
        
        %%

        %Screen('Flip', w2.window,[],[],2);
        [~,~,flip_time] = Screen('Flip', w1.window,[],[],1);
        
        ftimes(joystick_count-1) = flip_time - start_time;
        
        Screen('Close', frame_tex);


        
    end
    
    %%
    
    %if movie has reached the end:
    if frame_tex == -1,
        if eyetracking,
            talk2tobii('EVENT','stop_movie',GetSecs);
        end
        end_time = GetSecs;
        break;
    end;
    
    
    
    [ keyIsDown, timeSecs, keyCode ] = KbCheck(intKeyboard);
    
    % if pause key is pressed:
    if (strcmpi(KbName(keyCode),keys.pausekey))
        
        Screen('PlayMovie', movie, 0); %stop movie playback
        
        start_pause = GetSecs;
        if eyetracking,
            talk2tobii('EVENT','pause_on',start_pause);
        end
        
        %in case of crash:
        end_time_temp = GetSecs;
        total_time_temp = end_time_temp - start_time - paused_time;
        
        % adjust subject-related eyetracking parameters (e.g., calibration, distance to screen), etc.
        if eyetracking,
            quit_flag = tobii_setup_subject(w1,w2,debugging_mode,pos,intKeyboard,infant_mode);
            talk2tobii('START_TRACKING');
            talk2tobii('RECORD');
            WaitSecs(1); %not sure if necessary
            if quit_flag == 1,
                end_time = GetSecs;
                break;
            end
        elseif ~eyetracking
            pause;
        end
        
        %resume:
        Screen('PlayMovie', movie, 1, 0, 1.0); %start playback
        
        %account for start delay
        timeindex= Screen('GetMovieTimeIndex', movie);
        timeindex_old = timeindex;
        while timeindex == timeindex_old, %i.e., while frames haven't started appearing...
            timeindex = Screen('GetMovieTimeIndex', movie);
        end
        
        end_pause = GetSecs;
        if eyetracking,
            talk2tobii('EVENT','pause_off',end_pause);
        end
        
        paused_time_temp = end_pause - start_pause;
        paused_time = paused_time + paused_time_temp;
        
    end
    
    % if manually aborted via stopkey
    if (strcmpi(KbName(keyCode),keys.stopkey))
        end_time = GetSecs;
        if eyetracking,
            talk2tobii('EVENT','stop_movie',end_time);
        end
        disp('Experiment manually aborted.');
        break;
    end
    
    
    %to stop at a particular point in the movie, useful when debugging
    if GetSecs - start_time - paused_time >= manual_stop_time,
        if eyetracking,
            talk2tobii('EVENT','stop_movie',GetSecs);
        end
        end_time= GetSecs;
        break
    end;
    
    
%     if moviename_num < 3,
%         [ keyIsDown, timeSecs, keyCode ] = KbCheck(intKeyboard);
%         if keyIsDown && keyIsDown_prev ~= 1,
%             if (strcmpi(KbName(keyCode),keys.subj_response))
%                 subj_responses(end+1) = GetSecs - start_time;
%             end
%         end
%         keyIsDown_prev = keyIsDown;
%     end
    
end

%% Close down movie and eyetracking:

if eyetracking
    talk2tobii('STOP_RECORD');
    talk2tobii('STOP_TRACKING');
end


w1.subj_responses = subj_responses;

% w2.subj_responses = subj_responses;


%% verify validation at the end of the experiment

Beeper(1000,.1,.1);Beeper(1000,.1,.1);Beeper(1000,.1,.1);


if ~quit_flag
    Screen('PlayMovie', movie, 0);
    Screen('CloseMovie', movie);
end


if eyetracking
    quit_flag = tobii_end_validation(w1,w2,debugging_mode,pos,intKeyboard,infant_mode);
    if quit_flag,
        disp('manually aborted');
        return;
    end
end
    
    
if eyetracking,     
    talk2tobii('SAVE_DATA',[outdir1 outfile(1:end-4) '_TRACKING.txt'],[outdir1 outfile(1:end-4) '_EVENTS.txt'],'APPEND');
end


total_time = end_time - start_time - paused_time;

w1.total_time = total_time;
% w2.total_time = total_time;
if gamepad,
    w1.gamepad_out = gamepad_out;
%     w2.gamepad_out = gamepad_out;
%     Gamepad('Unplug');
end

w1.frame_times = ftimes;



% if ~strcmpi(task_str,'joystick_test'),

save([outdir1 outfile],'w1')

% save([outdir2 outfile],'w1')

% end

% save([outdir1 outfile],'w1','w2')

% WaitSecs(.2);

% Screen('Close', w1.window);
clear screen
ShowCursor;
% ListenChar(0);
disp(['total_time = ' num2str(total_time)]);

clear all

return

