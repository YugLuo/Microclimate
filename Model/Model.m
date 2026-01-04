%% PART B: Simulating Under-Canopy TMP

load '/ampha/tenant/fafu/private/user/guanyl/lyg/data/00Yugeng/01TrainedModel/Model_daily.mat'

[aspect, ref] = readgeoraster('/ampha/tenant/fafu/private/user/guanyl/lyg/data/00Yugeng/02Data/blocks/SRTM/ASP.tif');
[elevation, ~] = readgeoraster('/ampha/tenant/fafu/private/user/guanyl/lyg/data/00Yugeng/02Data/blocks/SRTM/ELEV.tif');
[slope, ~] = readgeoraster('/ampha/tenant/fafu/private/user/guanyl/lyg/data/00Yugeng/02Data/blocks/SRTM/SLP.tif');
[canopyHeight, ~] = readgeoraster('/ampha/tenant/fafu/private/user/guanyl/lyg/data/00Yugeng/02Data/blocks/CHT/CHT.tif');
[leafAreaIndex, ~] = readgeoraster('/ampha/tenant/fafu/private/user/guanyl/lyg/data/00Yugeng/02Data/blocks/LAI/A2000M1.tif');
[fractionAbsorbedRadiation, ~] = readgeoraster('/ampha/tenant/fafu/private/user/guanyl/lyg/data/00Yugeng/02Data/blocks/FAPAR/A2000M1.tif');
[monthlyPrecipitation, ~] = readgeoraster('/ampha/tenant/fafu/private/user/guanyl/lyg/data/00Yugeng/02Data/blocks/PCP/PCP2000M1.tif');
[monthlySolarRadiation, ~] = readgeoraster('/ampha/tenant/fafu/private/user/guanyl/lyg/data/00Yugeng/02Data/blocks/SLR/SLR2000M1.tif');
[monthlyOpenAirTemp, ~] = readgeoraster('/ampha/tenant/fafu/private/user/guanyl/lyg/data/00Yugeng/02Data/blocks/TMP/TMP2000M1.tif');

[row, col] = size(elevation);

predictors = cat(2, aspect(:), elevation(:), slope(:), canopyHeight(:), leafAreaIndex(:), fractionAbsorbedRadiation(:), monthlyPrecipitation(:), monthlySolarRadiation(:), monthlyOpenAirTemp(:));

predictedTempUnderCanopy = predict(Mdl, predictors);

reshapedPredictedTempUnderCanopy = single(reshape(predictedTempUnderCanopy,[row,col]));

outputFilePath = ['/ampha/tenant/fafu/private/user/guanyl/lyg/data/00Yugeng/03Understory_tmp/1/2000M1.tif'];

geotiffwrite(outputFilePath, reshapedPredictedTempUnderCanopy, ref,'TiffType','bigtiff')

