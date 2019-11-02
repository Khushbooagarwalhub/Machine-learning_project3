clear
clc

%% LOAD MAT OBJECT
struc = load('matlab_annotation.mat');
ann = struc.annotation;

%% ASSEMBLE IMAGE IDS, IMAGE NAMES, AND ACTIVITES IDS IN A CSV
% Extract info from struct and append to an array
images_info = ["Filename", "Action", "Action ID", "x1", "y1", "x2", "y2";];

idx = -1;
actions = string([]);
for i = 1:size(ann,2)
    img = ann(i);
    
    filename = string(img{1}.imageName);
    filename_list = split(filename, ["0", "1", "2", "3"]);
    action = filename_list(1);
    
    if ~any(actions == action)
        actions = [actions, action];
        idx = idx+1;
    end
    
    info = [filename, action, idx, img{1}.bbox];
    images_info = [images_info; info];
end
% Export array to a csv
file = "..\Resources\Images\image_info.csv";
writematrix(images_info, file)