function   [XYZ_refined] = matchXYZ(Points_GT,Points_in)

x1  = -100;
x2  = 100;
x1d = 5;
y1  = -100;
y2  = 100;
y1d = 5;
z1  = -300;
z2  = 300;
z1d = 5;

x2d = 0.1;
y2d = 0.1;
z2d = 0.1;

% Rough Matching Loop
out=nan(203401,4);
k=1;
for x=x1:x1d:x2
for y=y1:y1d:y2
    for z=z1:z1d:z2
        
        dist=pdist2(Points_GT,Points_in + [x y z]);
        out(k,:) = [[x y z],mean(dist(:))];
        k=k+1;
    end
end
end

[~,pos] = min(out(:,4));
XYZ_rough = out(pos,1:3);
disp(XYZ_rough);

k=0;
for x=XYZ_rough(1)-x1d:x2d:XYZ_rough(1)+x1d
    for y=XYZ_rough(2)-y1d:y2d:XYZ_rough(2)+y1d
        for z=XYZ_rough(3)-z1d:z2d:XYZ_rough(3)+z1d
            k=k+1;
        end
    end
end
% Refined Matching Loop
out=nan(k,4);
k=1;
for x=XYZ_rough(1)-x1d:x2d:XYZ_rough(1)+x1d
for y=XYZ_rough(2)-y1d:y2d:XYZ_rough(2)+y1d
    for z=XYZ_rough(3)-z1d:z2d:XYZ_rough(3)+z1d
        dist=pdist2(Points_GT,Points_in + [x y z]);
        out(k,:) = [[x y z],mean(dist(:))];
        k=k+1;
    end
end
end

[~,pos] = min(out(:,4));
XYZ_refined = out(pos,1:3);
disp(XYZ_refined);