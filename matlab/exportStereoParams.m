clear;
load('calibrationSession');
points1 = calibrationSession.BoardSet.BoardPoints(:,:,:,1);
points2 = calibrationSession.BoardSet.BoardPoints(:,:,:,2);

for i = 1:calibrationSession.BoardSet.NumBoards
    T = table();
    T.img_pts1 = points1(:,:,i);
    T.img_pts2 = points2(:,:,i);
    T.obj_pts = calibrationSession.BoardSet.WorldPoints;
    jsonstr = jsonencode(T);
    fid = fopen(sprintf('./jsonout/file_%03d.json', i), 'wt');
    fprintf(fid, jsonstr);
end