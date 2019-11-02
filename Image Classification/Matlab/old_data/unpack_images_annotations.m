% %% LOAD MAT OBJECT
% struc = load('images_annotations.mat');
% ann = struc.RELEASE;

%% ASSEMBLE IMAGE IDS, IMAGE NAMES, AND ACTIVITES IDS IN A CSV
% Extract info from struct and append to an array
images_info = ["ID", "Filename", "Act ID";];
count = 0;
for i = 1:size(ann.annolist,2)
    image = [string(i), string(ann.annolist(i).image.name), string(ann.act(i).act_id)];
    images_info = [images_info; image];
    if ann.act(i).act_id == -1
        count = count+1;
    end
end
% Export array to a csv
file = "..\Resources\Images\image_info.csv";
writematrix(images_info, file)

% %% EXTRACT ACTIVITY LABELS
% % Activity label are stored in a struct with as many rows as are images.
% % Consequently, some activities are repeated. 
% % We only want unique activities, so need to filter the struct. 
% 
% activity_ids = [];
% activities = ["ID", "Activity", "Category";];
% 
% for j = 1:size(ann.act, 1)
%     activity_id = ann.act(j).act_id;
%     % Skip this row if we've already inlcuded this activity in the array.
%     if (any(activity_ids == activity_id))
%         continue;
%     else
%     % Otherwise, add activity to array.
%         if string(ann.act(j).cat_name) == "inactivity quiet/light"
%             ann.act(j).cat_name = "inactivity quiet OR light";
%         end
%         
%         if string(ann.act(j).act_name) == "picking fruits/vegetables"
%             ann.act(j).act_name = "picking fruits and vegetables";
%         end
%         
%         if string(ann.act(j).cat_name) == "gardening, using containers, older adults > 60 years"
%             ann.act(j).cat_name = "gardening, using containers, older adults (over 60)";
%         end
%         
%         activity = [string(activity_id), ... 
%             string(ann.act(j).act_name), ...
%             string(ann.act(j).cat_name)];
%         activities = [activities; activity];
%         
%         activity_ids = [activity_ids, activity_id];
%     end 
% end
% % Export array to csv
% file1 = "..\Resources\Images\activities_info.csv";
% writematrix(activities, file1)