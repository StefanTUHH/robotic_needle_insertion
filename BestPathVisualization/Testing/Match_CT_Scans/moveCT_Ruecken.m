clear all
close all
clc

%% Get Files for Rueckenlage
files = dir('/home/Neidhardt/Dokumente/Paper_Neidhardt_Gerlach/Annotated_CTs_II/*/Ruecken*/*');
for f=1:length(files)
    folders{f} = [files(f).folder '/'];
end

folders_to_do = unique(folders);

% Get Annotaged Points and Names
for f=1:length(folders_to_do)
    AnnotationFiles{f} = dir([folders_to_do{f} '/*.mrk.json']);
end

% Get Volumes
for f=1:length(folders_to_do)
    FoldersVolume{f} = dir([folders_to_do{f} '/*.nrrd']);
end

%% Match to Target CT Scan -> Leiche 16 Ruecken, Leiche 17 Bauch
for p=1:16
    names_GT{p} = AnnotationFiles{8}(p).name;
    namePath = [AnnotationFiles{8}(p).folder '/' AnnotationFiles{8}(p).name];
    fid = fopen(namePath);
    raw = fread(fid,inf);
    str = char(raw');
    fclose(fid);
    val = jsondecode(str);
    Points_GT(p,:) = val.markups.controlPoints.position;
end

% Remove T_29 since random position sometimes
names_GT(14) = [];
Points_GT(14,:) = [];

for q = 1:length(AnnotationFiles)
    % Get Names of Points to Match
    names_IN = [];rm_idx=[];
    for p=1:length(AnnotationFiles{q})
        names_IN{p} = AnnotationFiles{q}(p).name;
        if contains(names_IN{p} ,'T_29')
            rm_idx = p;
        end
    end
    if ~isempty(rm_idx)
        names_IN(rm_idx)=[];
    end
    
    idx = ismember(names_GT,names_IN);
    disp([ names_GT(idx);names_IN]');
    idx_GT{q} = idx;
end

% Match Points
for q = 1:length(AnnotationFiles)
    Points_in = [];rm_idx=[];
    for p=1:length(AnnotationFiles{q})
        namePath = [AnnotationFiles{q}(p).folder '/' AnnotationFiles{q}(p).name];
        fid = fopen(namePath);
        raw = fread(fid,inf);
        str = char(raw');
        fclose(fid);
        val = jsondecode(str);
        Points_in(p,:) = val.markups.controlPoints.position;
        if contains(namePath ,'T_29')
            rm_idx = p;
        end
    end
    Points_in(rm_idx,:)=[];
    Points_GT(idx_GT{q},:);
    
    [XYZ] = matchXYZ(Points_GT,Points_in);
    t_CT(q,:) = XYZ;
end

%% Save to File
for vol_shift = 1:size(t_CT,1)
    str_V = [FoldersVolume{vol_shift}.folder];
    str_V = strrep(str_V,'Annotated_CTs_II','Colormaps_160');
    pos = strfind(str_V,'/');
    
    if contains(str_V,'Ruecken_I_')
        str_V = [str_V(1:pos(end)) 'Ruecken_I_XYZ.txt']
    elseif contains(str_V,'Ruecken_II_')
        str_V = [str_V(1:pos(end)) 'Ruecken_II_XYZ.txt'];
    else
        str_V = [str_V(1:pos(end)) 'Ruecken_XYZ.txt'];
    end
    
    t_vec = t_CT(vol_shift,:);
    
    writematrix(t_vec,str_V);
    
    str_V = strrep(str_V,'_160','_130');
    writematrix(t_vec,str_V);
end
%% Plot Colormaps Shifted

limCmap = 500;

for vol_shift = 1:size(t_CT,1)
    disp(vol_shift)
    t_vec = t_CT(vol_shift,:);
    
    str_V = [FoldersVolume{vol_shift}.folder '/'];
    str_V = strrep(str_V,'Annotated_CTs_II','Colormaps_160');
    file = dir([str_V 'T_13a*']);
    if isempty(file)
        file = dir([str_V '*.txt']);
        file = file(1);
    end
    points =  load([file.folder '/' file.name]);
    
    str_V_ref = [FoldersVolume{8}.folder '/'];
    str_V_ref = strrep(str_V_ref,'Annotated_CTs_II','Colormaps_160');
    file_ref = dir([str_V_ref 'T_13a*']);
	if isempty(file_ref)
        file_ref = dir([str_V_ref '*.txt']);
        file_ref = file_ref(1);
    end
    points_ref =  load([file_ref.folder '/' file_ref.name]);
    
    % Ref
    colormap_0 = points_ref(:,4,1);
    XYZ_colormap = points_ref(:,1:3,1);
    
    % Original
    c_jet = colormap('jet');
    skin = max(colormap_0);
    idx = find(colormap_0~=skin);
    
    % Limit Colormap
    idx_lim = find(colormap_0>limCmap);
    colormap_0(idx_lim) = skin;
    idx = find(colormap_0~=skin);
    
    c = (colormap_0(idx)+1024)/2048;   % Fixed Colorbar
    
    c = round(c*(length(c_jet)-1))+1;
    cjet = ones(length(colormap_0),1);
    cjet(idx) = c;
    c_jet(1,:) = [68,59,49]/100;
    ptCloud_0 = pointCloud(XYZ_colormap(:,1:3));
    cjet(cjet>length(c_jet)) = length(c_jet);
    ptCloud_0.Color = uint8(c_jet(cjet,:)*256);
    
    % Moved
    colormap_1 = points(:,4,1);
    XYZ_colormap_1 = points(:,1:3,1) + t_vec;
    
    % Original
    c_jet = colormap('jet');
    skin = max(colormap_1);
    idx = find(colormap_1~=skin);
    
    % Limit Colormap
    idx_lim = find(colormap_1>limCmap);
    colormap_1(idx_lim) = skin;
    idx = find(colormap_1~=skin);
    
    c = (colormap_1(idx)+1024)/2048;   % Fixed Colorbar
    
    c = round(c*(length(c_jet)-1))+1;
    cjet = ones(length(colormap_1),1);
    cjet(idx) = c;
    c_jet(1,:) = [68,59,49]/100;
    ptCloud_1 = pointCloud(XYZ_colormap_1(:,1:3));
    cjet(cjet>length(c_jet)) = length(c_jet);
    ptCloud_1.Color = uint8(c_jet(cjet,:)*256);
    
    
    fig = figure(1);clf;
    set(fig,'Position',[675 547 1029 415]);
    subplot 121
    pcshow(ptCloud_0);
    set(gcf,'color','w');
    set(gca,'color','w');
    set(gca, 'XColor', [0.15 0.15 0.15], 'YColor', [0.15 0.15 0.15], 'ZColor', [0.15 0.15 0.15])
    view(160,0)
    cb = colorbar;
    cb.Ticks =  linspace(0,1,9);
    cb.TickLabels = num2str(linspace(-1024,1024,9)');
    ylabel(cb,'Max HU');
    axis equal
    zlim([0 900])
    xlim([-550 100])
    
    subplot 122
    pcshow(ptCloud_1);
    set(gcf,'color','w');
    set(gca,'color','w');
    set(gca, 'XColor', [0.15 0.15 0.15], 'YColor', [0.15 0.15 0.15], 'ZColor', [0.15 0.15 0.15])
    view(160,0)
    cb = colorbar;
    cb.Ticks =  linspace(0,1,9);
    cb.TickLabels = num2str(linspace(-1024,1024,9)');
    ylabel(cb,'Max HU');
    axis equal
    zlim([0 900])
    xlim([-550 100])
    
    target=file.name;
    str = str_V(51:end-1);str=strrep(str,'/','_');str=[str '_' target(1:end-4) '.png'];
    print(['/home/Neidhardt/Dokumente/Paper_Neidhardt_Gerlach/Colormaps_Images/' str],'-r200','-dpng');
end

% skin = max(colormap_0);
% idx = find(colormap_0~=skin);