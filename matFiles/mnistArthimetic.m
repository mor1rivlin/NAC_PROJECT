close all

[imgDataTrain, labelsTrain, imgDataTest, labelsTest] = prepareData;
nTrainImg = 100000;
imgDataTrainArith = zeros(28,28*3,nTrainImg);
labelTr = zeros(nTrainImg,1);
r = randi(length(labelsTrain),nTrainImg,1);
r2 = randi(length(labelsTrain),nTrainImg,1);
for i=1:nTrainImg
    imgDataTrainArith(:,1:28,i) =double(imgDataTrain(:,:,1,r2(i)))/255.0;
    imgDataTrainArith(:,28+1:28*2,i) = plus;
    imgDataTrainArith(:,28*2+1:end,i) = double(imgDataTrain(:,:,1,r(i)))/255.0;
    labelTr(i) = double(labelsTrain(r2(i)))+double(labelsTrain(r(i)));
end
% for i=1:10
%     figure;
%     imshow(imgDataTrainArith(:,:,i))
%     title(num2str(labelTr(i)))
% end

nTestImg = 10000;
labelTes = zeros(nTestImg,1);
imgDataTestArith = zeros(28,28*3,nTestImg);
r = randi(length(labelsTest),nTestImg,1);
r2 = randi(length(labelsTest),nTestImg,1);

for i=1:nTestImg
    imgDataTestArith(:,1:28,i) = double(imgDataTest(:,:,1,r(i)))/255.0;
    imgDataTestArith(:,28+1:28*2,i) = plus;
    imgDataTestArith(:,28*2+1:end,i) = double(imgDataTest(:,:,1,r2(i)))/255.0;
    labelTes(i) = double(labelsTest(r2(i)))+double(labelsTest(r(i)));
end
figure;
for i=1:10
    subplot(2,5,i)
    imshow(imgDataTestArith(:,:,i))
    title(num2str(labelTes(i)-2))
end

% NormMean = mean(mean(mean(imgDataTestArith(:,:,1,:))));

% save('MnistData2.mat','imgDataTestArith','imgDataTrainArith','labelTes','labelTr','-v7.3')